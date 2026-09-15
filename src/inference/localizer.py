import os
import cv2
import sqlite3
import numpy as np

from pathlib import Path
from omegaconf import DictConfig

from src.preprocessing.preprocess import preprocess
from src.utils.model import create_model
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.decorators import performance
from src.utils.distance import haversine_distance

import faiss # NEED THIS BELOW TORCH IMPORTS!!


# FAISS + PyTorch error otherwise bc of OpenMP versioning
# See: https://github.com/ultralytics/yolov5/issues/5086
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Localizer:
    """Class to localize position."""
    def __init__(self, config: DictConfig, device: str):
        self.device = device
        self.model = create_model(config, device)
        self.k = config.inference.top_k
        self.dist_epsilon = config.inference.dist_epsilon
        self.score_epsilon = config.inference.score_epsilon
        self.beta = config.inference.beta
        self.anchor_bonus_weight = config.inference.anchor_bonus_weight

        faiss_path = Path(config.output.faiss)
        db_path = Path(config.output.db)

        if not faiss_path.exists() or not db_path.exists():
            preprocess(config)

        self.index = faiss.read_index(str(faiss_path))
        self.db_connection = sqlite3.connect(db_path)

    @performance
    def predict(self, image: np.ndarray) -> list[tuple[int, float, tuple[float, float]]]:
        """Predicts passed frame image"""
        embedding = self.model.embed_image(image)
        faiss.normalize_L2(embedding)
        scores, uids = self.index.search(embedding, self.k)

        res = []

        for uid, score in zip(uids[0], scores[0]):
            coords = self.get_coords(int(uid))
            res.append((
                int(uid),
                float(score),
                coords
            ))

        return res

    def estimate_position(self, predictions: list[tuple]) -> tuple[float, float]:
        """Based on the top k predictions, make a location estimate"""
        anchor = predictions[0][2]
        candidates = []

        for uid, score, coords in predictions:
            dist = haversine_distance(anchor, coords)
            if dist > self.dist_epsilon or score < self.score_epsilon:
                continue

            candidates.append((score, coords))

        scores = np.array([score for score, _ in candidates])

        logits = self.beta * (scores - scores.max())
        logits[0] += self.anchor_bonus_weight

        weights = np.exp(logits)    # softmax weighting 
        weights /= weights.sum()

        weighted_lon = 0.0
        weighted_lat = 0.0

        for weight, (_, (lon, lat)) in zip(weights, candidates):
            weighted_lon += weight * lon
            weighted_lat += weight * lat

        return float(weighted_lon), float(weighted_lat)

    def get_coords(self, uid: int) -> tuple:
        """Helper to fetch coords from db."""
        cursor = self.db_connection.cursor()

        cursor.execute("""
            SELECT center_lon, center_lat
            FROM tiles
            WHERE id = ?
            """, (int(uid),))

        return cursor.fetchone()


if __name__ == "__main__":
    config_path = Path("src/config/default.yaml")
    config = load_config(config_path)
    device = get_device()

    localizer = Localizer(config, device)

    image = cv2.imread("data/query4.png") # ex
    res = localizer.predict(image)
    print(res)

