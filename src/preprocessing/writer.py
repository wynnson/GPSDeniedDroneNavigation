import sqlite3
import faiss
import numpy as np

from pathlib import Path


class TileWriter:
    def __init__(
        self,
        db_path: Path,
        faiss_path: Path,
        embedding_dim: int
    ):
        self.db_path = db_path
        self.faiss_path = faiss_path

        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tiles (
                id INTEGER PRIMARY KEY,
                lon_left REAL,
                lat_bottom REAL,
                lon_right REAL,
                lat_top REAL
            )
        """)

        self.cursor.execute("SELECT COALESCE(MAX(id), -1) FROM tiles")
        self.next_id = self.cursor.fetchone()[0] + 1

        if faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)

    def write_batch(
        self,
        metadata_batch: list[dict],
        embeddings: np.ndarray,
    ):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        rows = []

        for metadata in metadata_batch:
            rows.append((
                self.next_id,
                metadata["lon_left"],
                metadata["lat_bottom"],
                metadata["lon_right"],
                metadata["lat_top"],
            ))

            self.next_id += 1

        self.cursor.executemany("""
            INSERT OR IGNORE INTO tiles (
                id,
                lon_left,
                lat_bottom,
                lon_right,
                lat_top
            )
            VALUES (?, ?, ?, ?, ?)
        """, rows)

        self.index.add(embeddings)

    def close(self):
        self.connection.commit()
        faiss.write_index(self.index, str(self.faiss_path))
        self.cursor.close()
        self.connection.close()
