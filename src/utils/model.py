import torch

from torch import nn
from omegaconf import DictConfig

from src.models.dinov2 import DinoV2
from src.models.base import PreprocessingModel


def load_model(config: DictConfig, device: str) -> nn.Module:
    """Loads model based on config"""
    if config.model.source == "torch_hub":
        if not config.model.repo or not config.model.name:
            raise ValueError("Missing torch hub configuration")

        model = torch.hub.load(config.model.repo, config.model.name)

    else:
        # TODO: Future endeavor for own models etc.
        pass

    if model is None:
        raise ValueError("Missing model")

    model.to(device)
    model.eval()
    return model


def create_model(config: DictConfig, device: str) -> PreprocessingModel:
    raw_model = load_model(config, device)

    if config.model.type == "dinov2":
        return DinoV2(model=raw_model, device=device)

    raise ValueError(f"Unsupported model: {config.model.type}")
