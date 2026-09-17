from pathlib import Path

import torch

from omegaconf import DictConfig

from src.models.dinov2_onnx import DinoV2_ONNX
from src.models.dinov2 import DinoV2
from src.models.base import InferenceModel, PreprocessingModel


def load_model_torch_hub(config: DictConfig, device: str) -> PreprocessingModel:
    """Loads model from torch hub based on config"""
    if not config.model.repo or not config.model.name:
        raise ValueError("Missing torch hub configuration")

    raw_model = torch.hub.load(config.model.repo, config.model.name)
    raw_model.to(device)
    raw_model.eval()

    match config.model.type:
        case "dinov2":
            model = DinoV2(model=raw_model, device=device)
        case _:
            raise ValueError(f"Unsupported model type: {config.model.type}")

    return model


def load_model_onnx(config: DictConfig) -> InferenceModel:
    """Loads ONNX model from disk"""
    model_path = config.model.file_path
    if not model_path or not Path(model_path).exists():
        raise ValueError(f"File path does not exist: {model_path}")

    match config.model.type:
        case "dinov2":
            model = DinoV2_ONNX(model_path)
        case _:
            raise ValueError(f"Unsupported model type: {config.model.type}")

    return model


def create_model(config: DictConfig, device: str) -> PreprocessingModel | InferenceModel:
    if not config.model.type:
        raise ValueError(f"Unsupported model type: {config.model.type}")

    match config.model.source:
        case "torch_hub":
            model = load_model_torch_hub(config, device)
        case "onnx":
            model = load_model_onnx(config)
        case _:
            raise ValueError(f"Unsupported source: {config.model.source}")

    return model
