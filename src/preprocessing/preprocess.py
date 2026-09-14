from pathlib import Path
from omegaconf import DictConfig

from src.preprocessing.raster_splitter import embed_raster
from src.preprocessing.writer import TileWriter
from src.utils.device import get_device
from src.utils.config import load_config
from src.utils.model import create_model
from src.utils.decorators import performance


@performance
def preprocess(config: DictConfig):
    """Runs the preprocessing pipeline."""
    device = get_device()
    model = create_model(config, device)

    writer = TileWriter(
        db_path=config.output.db,
        faiss_path=config.output.faiss,
        embedding_dim=config.model.embedding_dim
    )

    try:
        embed_raster(
            file_path=config.raster.file_path,
            model=model,
            writer=writer,
            window_size=config.raster.window_size,
            batch_size=config.raster.batch_size,
            dst_crs=config.raster.dst_crs,
        )

    finally:
        writer.close()


if __name__ == "__main__":
    config_path = Path("src/config/default.yaml")
    config = load_config(config_path)
    preprocess(config)
