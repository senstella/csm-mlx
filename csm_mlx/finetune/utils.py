# Copyright © 2024 Apple Inc. - https://github.com/ml-explore/mlx-lm/blob/3b3df251d349f2322deeacaf28152a97c0dcd8dd/mlx_lm/tuner/utils.py
import json
import types
from pathlib import Path
from typing import Dict

import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

from csm_mlx.models import CSM


def linear_to_lora_layers(
    model: nn.Module,
    config: Dict,
    use_dora: bool = False,
):
    """
    Convert some of the models linear layers to lora layers.

    Args:
        model (nn.Module): The neural network model.
        config (dict): More configuration parameters for LoRA, including the
          rank, scale, and optional layer keys.
        use_dora (bool): If True, uses DoRA instead of LoRA.
          Default: ``False``
    """

    def to_lora(layer):
        if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
            LoRALayer = DoRALinear if use_dora else LoRALinear
        elif isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
            if use_dora:
                raise ValueError(f"{type(layer).__name__} doesn't support DoRA yet.")
            LoRALayer = LoRASwitchLinear
        elif isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)):
            LoRALayer = DoRAEmbedding if use_dora else LoRAEmbedding
        else:
            raise ValueError(
                f"Can't convert layer of type {type(layer).__name__} to LoRA"
            )

        return LoRALayer.from_base(  # type: ignore
            layer,  # type: ignore
            r=config["rank"],
            scale=config["scale"],
            dropout=config["dropout"],
        )

    keys: set[str] | None = config.get("keys", None)
    if keys is not None:
        keys = set(keys)
    else:
        keys = set()

    if "attn" in keys and getattr(model, "model_type", None) in [
        "llama",
    ]:  # Support attn params!
        keys.update(["self_attn.q_proj", "self_attn.v_proj"])

    if isinstance(model, CSM):  # Recursively search for backbone and decoder
        linear_to_lora_layers(model.backbone, config, use_dora)
        linear_to_lora_layers(model.decoder, config, use_dora)

    for layer in getattr(model, "layers", []):  # type: ignore
        lora_layers = [(k, to_lora(m)) for k, m in layer.named_modules() if k in keys]
        if lora_layers:
            layer.update_modules(tree_unflatten(lora_layers))

    lora_modules = [(k, to_lora(m)) for k, m in model.named_modules() if k in keys]
    if lora_modules:
        model.update_modules(tree_unflatten(lora_modules))


def load_adapters(model: nn.Module, adapter_path: str) -> nn.Module:
    """
    Load any fine-tuned adapters / layers.

    Args:
        model (nn.Module): The neural network model.
        adapter_path (str): Path to the adapter configuration file.

    Returns:
        nn.Module: The updated model with LoRA layers applied.
    """
    _adapter_path = Path(adapter_path)
    if not _adapter_path.exists():
        raise FileNotFoundError(f"The adapter path does not exist: {_adapter_path}")
    with open(_adapter_path / "adapter_config.json", "r") as fid:
        config = types.SimpleNamespace(**json.load(fid))
    fine_tune_type = getattr(config, "fine_tune_type", "lora")
    if fine_tune_type != "full":
        linear_to_lora_layers(
            model,
            config.lora_parameters,
            use_dora=(fine_tune_type == "dora"),
        )
    model.load_weights(str(_adapter_path / "adapters.safetensors"), strict=False)
    return model
