import torch

from torch import nn
from omegaconf import DictConfig


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
