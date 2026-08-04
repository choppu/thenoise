import os
import re
from typing import Dict, List, Optional, Tuple, Union
import torch
from tqdm import tqdm
from thenoise.utils.device import synchronize_device
# fp8 dropped: bf16-only engine

from thenoise.utils.safetensors import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks, get_split_weight_filenames

# NOTE (pruned for the focused engine): upstream merged LoHa/LoKr LoRAs via the
# `networks` package, which pulls in the whole sd-scripts LoRA/training stack. That
# package is not vendored; only standard LoRA (lora_down/lora_up) is merged here, and
# LoHa/LoKr raise NotImplementedError in the merge path.

from thenoise.utils.setup_logging import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _match_lora_keys(
    model_weight_key: str,
    lora_weight_keys: set,
) -> Optional[Tuple[str, str, str]]:
    """Find matching LoRA down/up/alpha keys for a model weight key.

    Returns (down_key, up_key, alpha_key) or None if no match.
    """
    if not model_weight_key.endswith(".weight"):
        return None

    lora_name_without_prefix = model_weight_key.rsplit(".", 1)[0]

    # sd-scripts naming: underscore-joined path, lora_down/lora_up.
    for prefix in ["lora_unet_", ""]:
        lora_name = prefix + lora_name_without_prefix.replace(".", "_")
        down_key = lora_name + ".lora_down.weight"
        up_key = lora_name + ".lora_up.weight"
        alpha_key = lora_name + ".alpha"
        if down_key in lora_weight_keys and up_key in lora_weight_keys:
            return (down_key, up_key, alpha_key)

    # diffusers-style naming: dotted path, lora_A/lora_B.
    for prefix in ["diffusion_model.", ""]:
        lora_name = prefix + lora_name_without_prefix
        a_key = lora_name + ".lora_A.weight"
        b_key = lora_name + ".lora_B.weight"
        alpha_key = lora_name + ".alpha"
        if a_key in lora_weight_keys and b_key in lora_weight_keys:
            return (a_key, b_key, alpha_key)

    return None


def _compute_lora_delta(
    model_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    alpha,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Compute the LoRA delta for a single weight: multiplier * (up @ down) * scale.

    Returns a tensor of the same shape as model_weight.
    """
    dim = down_weight.size()[0]
    if isinstance(alpha, torch.Tensor):
        scale = float(alpha.to(calc_device)) / dim
    else:
        scale = alpha / dim

    down_weight = down_weight.to(calc_device)
    up_weight = up_weight.to(calc_device)

    original_dtype = model_weight.dtype
    if original_dtype.itemsize == 1:  # fp8
        down_weight = down_weight.to(torch.float16)
        up_weight = up_weight.to(torch.float16)

    if len(model_weight.size()) == 2:
        # linear
        if len(up_weight.size()) == 4:
            up_weight = up_weight.squeeze(3).squeeze(2)
            down_weight = down_weight.squeeze(3).squeeze(2)
        delta = multiplier * (up_weight @ down_weight) * scale
    elif down_weight.size()[2:4] == (1, 1):
        # conv2d 1x1
        delta = (
            multiplier
            * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
            .unsqueeze(2).unsqueeze(3)
            * scale
        )
    else:
        # conv2d 3x3
        conved = torch.nn.functional.conv2d(
            down_weight.permute(1, 0, 2, 3), up_weight
        ).permute(1, 0, 2, 3)
        delta = multiplier * conved * scale

    if original_dtype.itemsize == 1:  # fp8
        delta = delta.to(original_dtype)

    return delta


def apply_lora_to_model(
    model: torch.nn.Module,
    lora_sds: List[Dict[str, torch.Tensor]],
    multipliers: List[float],
    calc_device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Apply LoRA weights directly to a model's parameters (in-place).

    Returns a dict of {param_key: delta_tensor} representing the deltas applied,
    which can be passed to ``undo_lora_on_model`` to restore the original weights.

    Param keys use the same naming as ``model.state_dict()`` (e.g. "blocks.0.attn.gate.weight").
    """
    if not lora_sds:
        return {}

    if multipliers is None:
        multipliers = [1.0] * len(lora_sds)
    while len(multipliers) < len(lora_sds):
        multipliers.append(1.0)
    multipliers = multipliers[: len(lora_sds)]

    logger.info("Applying LoRA to model. multipliers: %s", multipliers)

    # Build key sets for each LoRA
    lora_weight_keys_list = [set(sd.keys()) for sd in lora_sds]

    # Collect all model param keys that end with .weight
    state = model.state_dict()
    undo_deltas: Dict[str, torch.Tensor] = {}

    for model_key, model_weight in state.items():
        if not model_key.endswith(".weight"):
            continue

        original_device = model_weight.device
        weight_on_calc = model_weight if original_device == calc_device else model_weight.to(calc_device)

        for lora_weight_keys, lora_sd, multiplier in zip(lora_weight_keys_list, lora_sds, multipliers):
            match = _match_lora_keys(model_key, lora_weight_keys)
            if match is None:
                continue

            down_key, up_key, alpha_key = match
            down_weight = lora_sd[down_key]
            up_weight = lora_sd[up_key]
            alpha = lora_sd.get(alpha_key, down_weight.size()[0])

            delta = _compute_lora_delta(weight_on_calc, down_weight, up_weight, alpha, multiplier, calc_device)

            # Accumulate delta (multiple LoRAs can affect the same layer)
            if model_key in undo_deltas:
                undo_deltas[model_key] = undo_deltas[model_key] + delta.to(undo_deltas[model_key].device, undo_deltas[model_key].dtype)
            else:
                undo_deltas[model_key] = delta

            # Remove consumed keys
            lora_weight_keys.remove(down_key)
            lora_weight_keys.remove(up_key)
            if alpha_key in lora_weight_keys:
                lora_weight_keys.remove(alpha_key)

    # Warn about unused LoRA keys
    for i, lora_weight_keys in enumerate(lora_weight_keys_list):
        if len(lora_weight_keys) > 0:
            logger.warning("LoRA %d has unused keys: %s", i, ", ".join(list(lora_weight_keys)[:10]))

    # Apply accumulated deltas to model parameters (in-place)
    if undo_deltas:
        with torch.no_grad():
            for param_key, delta in undo_deltas.items():
                param = model.get_submodule(".".join(param_key.split(".")[:-1])) if "." in param_key else model
                param_name = param_key.rsplit(".", 1)[-1]
                # Navigate to the module and apply delta
                # Simpler: use state_dict approach
                state[param_key] = state[param_key] + delta.to(
                    state[param_key].device, state[param_key].dtype
                )
        model.load_state_dict(state)

    # Move deltas to match param devices for efficient undo
    for key in undo_deltas:
        undo_deltas[key] = undo_deltas[key].to(calc_device)

    return undo_deltas


def undo_lora_on_model(
    model: torch.nn.Module,
    undo_deltas: Dict[str, torch.Tensor],
    calc_device: torch.device,
) -> None:
    """Undo a previous LoRA application by subtracting the stored deltas.

    Restores the model's parameters to their pre-LoRA state (in-place).
    """
    if not undo_deltas:
        return

    logger.debug("Undoing LoRA on model (%d keys)", len(undo_deltas))
    with torch.no_grad():
        state = model.state_dict()
        for param_key, delta in undo_deltas.items():
            state[param_key] = state[param_key] - delta.to(
                state[param_key].device, state[param_key].dtype
            )
        model.load_state_dict(state)


# ---------------------------------------------------------------------------
# Legacy load-time merge (kept for backward compatibility during transition)
# ---------------------------------------------------------------------------

def filter_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    # apply include/exclude patterns
    original_key_count = len(weights_sd.keys())
    if include_pattern is not None:
        regex_include = re.compile(include_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
        logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")

    if exclude_pattern is not None:
        original_key_count_ex = len(weights_sd.keys())
        regex_exclude = re.compile(exclude_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
        logger.info(f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}")

    if len(weights_sd) != original_key_count:
        remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
        remaining_keys.sort()
        logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
        if len(weights_sd) == 0:
            logger.warning("No keys left after filtering.")

    return weights_sd


def load_safetensors_with_lora(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]],
    lora_multipliers: Optional[List[float]],
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model.

    Args:
        model_files (Union[str, List[str]]): Path to the model file or list of paths. If the path matches a pattern like `00001-of-00004`, it will load all files with the same prefix.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): List of dictionaries of LoRA weight tensors to load.
        lora_multipliers (Optional[List[float]]): List of multipliers for LoRA weights.
        calc_device (torch.device): Device to calculate on.
        move_to_device (bool): Whether to move tensors to the calculation device after loading.
        dit_weight_dtype (Optional[torch.dtype]): Dtype to load weights in.
        disable_numpy_memmap (bool): Whether to disable numpy memmap when loading safetensors.
        weight_transform_hooks (Optional[WeightTransformHooks]): Hooks for transforming weights during loading.
    """

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    if isinstance(model_files, str):
        model_files = [model_files]

    extended_model_files = []
    for model_file in model_files:
        split_filenames = get_split_weight_filenames(model_file)
        if split_filenames is not None:
            extended_model_files.extend(split_filenames)
        else:
            extended_model_files.append(model_file)
    model_files = extended_model_files
    logger.info(f"Loading model files: {model_files}")

    # load LoRA weights
    weight_hook = None
    if lora_weights_list is None or len(lora_weights_list) == 0:
        lora_weights_list = []
        lora_multipliers = []
        list_of_lora_weight_keys = []
    else:
        list_of_lora_weight_keys = []
        for lora_sd in lora_weights_list:
            lora_weight_keys = set(lora_sd.keys())
            list_of_lora_weight_keys.append(lora_weight_keys)

        if lora_multipliers is None:
            lora_multipliers = [1.0] * len(lora_weights_list)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)
        if len(lora_multipliers) > len(lora_weights_list):
            lora_multipliers = lora_multipliers[: len(lora_weights_list)]

        # Merge LoRA weights into the state dict
        logger.info(f"Merging LoRA weights into state dict. multipliers: {lora_multipliers}")

        # make hook for LoRA merging
        def weight_hook_func(model_weight_key, model_weight: torch.Tensor, keep_on_calc_device=False):
            nonlocal list_of_lora_weight_keys, lora_weights_list, lora_multipliers, calc_device

            if not model_weight_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            if original_device != calc_device:
                model_weight = model_weight.to(calc_device)  # to make calculation faster

            for lora_weight_keys, lora_sd, multiplier in zip(list_of_lora_weight_keys, lora_weights_list, lora_multipliers):
                # check if this weight has LoRA weights
                lora_name_without_prefix = model_weight_key.rsplit(".", 1)[0]  # remove trailing ".weight"
                found = False

                # sd-scripts naming: underscore-joined path, lora_down/lora_up.
                # e.g. model "blocks.0.attn.gate.weight" <-> LoRA
                # "lora_unet_blocks_0_attn_gate.lora_down.weight".
                for prefix in ["lora_unet_", ""]:
                    lora_name = prefix + lora_name_without_prefix.replace(".", "_")
                    down_key = lora_name + ".lora_down.weight"
                    up_key = lora_name + ".lora_up.weight"
                    alpha_key = lora_name + ".alpha"
                    if down_key in lora_weight_keys and up_key in lora_weight_keys:
                        found = True
                        break

                if not found:
                    # diffusers-style naming: dotted path matching the model key (with a
                    # "diffusion_model." prefix), lora_A/lora_B instead of lora_down/lora_up.
                    # e.g. model "txtfusion.layerwise_blocks.0.mlp.gate.weight" <-> LoRA
                    # "diffusion_model.txtfusion.layerwise_blocks.0.mlp.gate.lora_A.weight".
                    for prefix in ["diffusion_model.", ""]:
                        lora_name = prefix + lora_name_without_prefix
                        a_key = lora_name + ".lora_A.weight"
                        b_key = lora_name + ".lora_B.weight"
                        alpha_key = lora_name + ".alpha"
                        if a_key in lora_weight_keys and b_key in lora_weight_keys:
                            down_key, up_key = a_key, b_key
                            found = True
                            break

                if found:
                    # Standard LoRA merge
                    # get LoRA weights
                    down_weight = lora_sd[down_key]
                    up_weight = lora_sd[up_key]

                    dim = down_weight.size()[0]
                    alpha = lora_sd.get(alpha_key, dim)
                    scale = alpha / dim

                    down_weight = down_weight.to(calc_device)
                    up_weight = up_weight.to(calc_device)

                    original_dtype = model_weight.dtype
                    if original_dtype.itemsize == 1:  # fp8
                        # temporarily convert to float16 for calculation
                        model_weight = model_weight.to(torch.float16)
                        down_weight = down_weight.to(torch.float16)
                        up_weight = up_weight.to(torch.float16)

                    # W <- W + U * D
                    if len(model_weight.size()) == 2:
                        # linear
                        if len(up_weight.size()) == 4:  # use linear projection mismatch
                            up_weight = up_weight.squeeze(3).squeeze(2)
                            down_weight = down_weight.squeeze(3).squeeze(2)
                        model_weight = model_weight + multiplier * (up_weight @ down_weight) * scale
                    elif down_weight.size()[2:4] == (1, 1):
                        # conv2d 1x1
                        model_weight = (
                            model_weight
                            + multiplier
                            * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                            * scale
                        )
                    else:
                        # conv2d 3x3
                        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                        # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                        model_weight = model_weight + multiplier * conved * scale

                    if original_dtype.itemsize == 1:  # fp8
                        model_weight = model_weight.to(original_dtype)  # convert back to original dtype

                    # remove LoRA keys from set
                    lora_weight_keys.remove(down_key)
                    lora_weight_keys.remove(up_key)
                    if alpha_key in lora_weight_keys:
                        lora_weight_keys.remove(alpha_key)
                    continue

                # Check for LoHa/LoKr weights with same prefix search
                for prefix in ["lora_unet_", ""]:
                    lora_name = prefix + lora_name_without_prefix.replace(".", "_")
                    hada_key = lora_name + ".hada_w1_a"
                    lokr_key = lora_name + ".lokr_w1"

                    if hada_key in lora_weight_keys or lokr_key in lora_weight_keys:
                        raise NotImplementedError(
                            "LoHa/LoKr LoRAs are not supported: the networks package "
                            "was not vendored. Only standard LoRA (lora_down/lora_up) is merged."
                        )

            if not keep_on_calc_device and original_device != calc_device:
                model_weight = model_weight.to(original_device)  # move back to original device
            return model_weight

        weight_hook = weight_hook_func

    state_dict = load_safetensors_with_lora_and_hook(
        model_files,
        calc_device,
        move_to_device,
        dit_weight_dtype,
        weight_hook=weight_hook,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=weight_transform_hooks,
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        # check if all LoRA keys are used
        if len(lora_weight_keys) > 0:
            # if there are still LoRA keys left, it means they are not used in the model
            # this is a warning, not an error
            logger.warning(f"Warning: not all LoRA keys are used: {', '.join(lora_weight_keys)}")

    return state_dict


def load_safetensors_with_lora_and_hook(
    model_files: list[str],
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    weight_hook: callable = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Load state dict from safetensors files and apply the optional LoRA merge hook.
    """
    logger.info(
        f"Loading state dict. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
    )
    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
            f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f
            for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False):
                if weight_hook is None and move_to_device:
                    value = f.get_tensor(key, device=calc_device, dtype=dit_weight_dtype)
                else:
                    value = f.get_tensor(key)  # we cannot directly load to device because get_tensor does non-blocking transfer
                    if weight_hook is not None:
                        value = weight_hook(key, value, keep_on_calc_device=move_to_device)
                    if move_to_device:
                        value = value.to(calc_device, dtype=dit_weight_dtype, non_blocking=True)
                    elif dit_weight_dtype is not None:
                        value = value.to(dit_weight_dtype)

                state_dict[key] = value
    if move_to_device:
        synchronize_device(calc_device)

    return state_dict
