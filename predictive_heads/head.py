from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class PredictiveHead(nn.Module, ABC):
    @abstractmethod
    def __init__(self, config):
        super().__init__()

    @abstractmethod
    def forward(self, seq_output) -> tuple[torch.Tensor, torch.Tensor]:
        """returns the start and end logits of the predictive head"""
        pass

PREDICTIVE_HEADS: dict[str, type[PredictiveHead]] = {}

def register_head(name: str):
    def decorator(cls: type[PredictiveHead]):
        PREDICTIVE_HEADS[name] = cls
        return cls
    return decorator

def get_pred_head(config):
    head_type = getattr(config, "head_type").lower()
    if head_type not in PREDICTIVE_HEADS:
        raise ValueError(f"Unknown head_type: '{head_type}")
    return PREDICTIVE_HEADS[head_type](config)
