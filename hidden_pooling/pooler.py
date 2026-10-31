from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class HiddenStatePooler(nn.Module, ABC):
    @abstractmethod
    def __init__(self, config):
        super().__init__()

    @abstractmethod
    def forward(self, full_hidden_states, attention_mask) -> torch.Tensor:
        """returns the accumulated hidden states across all layers"""
        pass

HIDDEN_POOLERS: dict[str, type[HiddenStatePooler]] = {}

def register_pooler(name: str):
    def decorator(cls: type[HiddenStatePooler]):
        HIDDEN_POOLERS[name] = cls
        return cls

    return decorator

def get_hidden_pooler(config):
    pooler_type = getattr(config, "pooler_type").lower()
    if pooler_type not in HIDDEN_POOLERS:
        raise ValueError(f"Unknown pooler_type: '{pooler_type}")
    return HIDDEN_POOLERS[pooler_type](config)
