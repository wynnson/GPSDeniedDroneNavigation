import os
import sqlite3
import numpy as np

from pathlib import Path

import faiss # NEED THIS BELOW ANY TORCH IMPORTS!! (Best to keep it on bottom)


# FAISS + PyTorch error otherwise bc of OpenMP versioning
# See: https://github.com/ultralytics/yolov5/issues/5086
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TileDatabaseManager:
    """
    Manager for db (FAISS and SQLlite).
    Important: Since this contains FAISS, it should always go last.
    Importing torch before FAISS causes segfaults.
    """
    def __init__(
        self,
        db_path: str,
        faiss_path: str,
        embedding_dim: int
    ):
        self.db_path = Path(db_path)
        self.faiss_path = Path(faiss_path)

        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tiles (
                id INTEGER PRIMARY KEY,
                center_lon REAL,
                center_lat REAL
            )
        """)

        self.cursor.execute("SELECT COALESCE(MAX(id), -1) FROM tiles")
        self.next_id = self.cursor.fetchone()[0] + 1

        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)

    def write_batch(
        self,
        metadata_batch: list[dict],
        embeddings: np.ndarray,
    ) -> None:
        """Writes metadata and embeddings to DB."""
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        rows = []

        for metadata in metadata_batch:
            rows.append((
                self.next_id,
                metadata["center_lon"],
                metadata["center_lat"],
            ))

            self.next_id += 1

        self.cursor.executemany("""
            INSERT OR IGNORE INTO tiles (
                id,
                center_lon,
                center_lat
            )
            VALUES (?, ?, ?)
        """, rows)

        self.index.add(embeddings)

    def get_coords(self, uid: int) -> tuple[float, float] | None:
        """Helper to fetch coords from db."""
        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT center_lon, center_lat
            FROM tiles
            WHERE id = ?
            """, (int(uid),))

        return cursor.fetchone()

    def search(self, embedding: np.ndarray, k: int) -> tuple:
        """Searches for similar embeddings"""
        faiss.normalize_L2(embedding)
        scores, uids = self.index.search(embedding, k)
        return scores, uids

    def close(self) -> None:
        """Cleans up connections."""
        self.connection.commit()
        faiss.write_index(self.index, str(self.faiss_path))
        self.cursor.close()
        self.connection.close()
