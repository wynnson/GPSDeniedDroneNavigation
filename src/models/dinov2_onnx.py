import cv2
import numpy as np
import onnxruntime as ort

from pathlib import Path

from src.models.base import InferenceModel


class DinoV2_ONNX(InferenceModel):
    def __init__(self, model_path: str | Path):
        self.size = (518, 518)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        image = self.normalize_image(image)

        embedding = self.session.run(
            [self.output_name],
            {self.input_name: image}
        )[0]

        return embedding
