import os
import re
from typing import Dict, List, Optional, Tuple, Union
import torch
from thenoise.utils.device import synchronize_device

from thenoise.utils.safetensors import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks, get_split_weight_filenames

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


def _unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a torch.compile OptimizedModule to get the original module.

    torch.compile wraps the model in an OptimizedModule whose state_dict()/load_state_dict()
    may not delegate correctly. Operating on the original module ensures LoRA key matching 
    and weight modification work correctly. The compiled kernels reference the same underlying 
    parameter tensors, so they see updates.
    """
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


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

    base_model = _unwrap_compiled(model)

    # Build key sets for each LoRA
    lora_weight_keys_list = [set(sd.keys()) for sd in lora_sds]

    # Collect all model param keys that end with .weight
    state = base_model.state_dict()
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
        base_model.load_state_dict(state)

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
    base_model = _unwrap_compiled(model)
    with torch.no_grad():
        state = base_model.state_dict()
        for param_key, delta in undo_deltas.items():
            state[param_key] = state[param_key] - delta.to(
                state[param_key].device, state[param_key].dtype
            )
        base_model.load_state_dict(state)

