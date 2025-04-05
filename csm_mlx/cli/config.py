from enum import Enum

from csm_mlx import csm_1b


class Models(str, Enum):
    _1b = "1b"


class OptimizerChoice(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"


MODEL = {
    "1b": {
        "loader": {
            "repo_id": "senstella/csm-1b-mlx",
            "filename": "ckpt.safetensors",
        },
        "config": csm_1b(),
        "sampling_rate": 24000,
    }
}
