"""LoRA network types.

Only standard LoRA (lora_down / lora_up) is merged natively in
``diffuse.utils.lora``. LoHa / LoKr require the full networks package
(``lora`` / ``network_arch``), which was not vendored upstream — they are
imported lazily in ``lora.py`` and will raise if a LoHa/LoKr LoRA is used.
Port the full package here when LoHa/LoKr support is wanted.
"""
