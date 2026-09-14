from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseModel(ABC):
    @abstractmethod
    def normalize_batch(self, tile_batch: list[np.ndarray], device: str) -> torch.Tensor:
        """Normalizes batch tiles"""
        pass

    @abstractmethod
    def embed_batch(self, tile_batch: list[np.ndarray]) -> np.ndarray:
        """Convert batch of tiles into embedding vector"""
        pass

    @abstractmethod
    def normalize_image(self, image: np.ndarray) -> torch.Tensor:
        """Normalizes individual image frames"""
        pass

    @abstractmethod
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Converts image into embedding vector"""
        pass
