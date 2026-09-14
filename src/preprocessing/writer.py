import sqlite3
import faiss
import numpy as np

from pathlib import Path


class TileWriter:
    """Helper class that performs IO operations on DB (SQL + FAISS)"""
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
                lon_left REAL,
                lat_bottom REAL,
                lon_right REAL,
                lat_top REAL
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

    def close(self) -> None:
        """Cleans up connections."""
        self.connection.commit()
        faiss.write_index(self.index, str(self.faiss_path))
        self.cursor.close()
        self.connection.close()
