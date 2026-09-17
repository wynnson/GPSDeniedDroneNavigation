from abc import ABC, abstractmethod

import numpy as np


class PreprocessingModel(ABC):
    @abstractmethod
    def embed_batch(self, tile_batch: list[np.ndarray]) -> np.ndarray:
        """Convert batch of tiles into embedding vector"""
        pass


class InferenceModel(ABC):
    """Mostly for ONNX inference."""
    @abstractmethod
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Converts image into embedding vector"""
        pass
