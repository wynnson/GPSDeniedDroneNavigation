import cv2
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from src.models.base import PreprocessingModel, InferenceModel


class DinoV2(PreprocessingModel, InferenceModel):
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
        self.size = (518, 518) # DINO v2 native
        
        # Imagenet defaults
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def normalize_batch(self, tile_batch: list[np.ndarray]) -> torch.Tensor:
        batch = torch.from_numpy(np.stack(tile_batch)).float().to(self.device) / 255.0
        batch = F.interpolate(
            batch,
            size=self.size,
            mode="bilinear",
            align_corners=False
        )
    
        return (batch - self.mean) / self.std

    def embed_batch(self, tile_batch: list[np.ndarray]) -> torch.Tensor:
        normalized = self.normalize_batch(tile_batch)

        with torch.inference_mode():
            embeddings = self.model(normalized)

        return embeddings.cpu().numpy()

    def normalize_image(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float().to(self.device) / 255.0
        image = (image - self.mean) / self.std
        return image

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        image = self.normalize_image(image)

        with torch.inference_mode():
            embedding = self.model(image)

        return embedding.cpu().numpy()
