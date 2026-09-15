import rasterio
import numpy as np

from collections.abc import Iterable
from rasterio.windows import Window, bounds
from rasterio.warp import transform_bounds
from pathlib import Path

from src.models.base import BaseModel
from src.preprocessing.writer import TileWriter


def iterate_raster(
    path: Path,
    stride: int,
    window_size: int,
    dst_crs: str = "EPSG:4326"
) -> Iterable[tuple[np.ndarray, dict]]:
    """
    Slides window through original raster, chopping it up
    into tiny pieces to embed.
    """
    try:
        with rasterio.open(path) as src:
            for r in range(0, src.height, stride):
                for c in range(0, src.width, stride):
                    width = min(window_size, src.width - c)
                    height = min(window_size, src.height - r)

                    if width < window_size or height < window_size:
                        continue

                    window = Window(c, r, width, height)

                    RGB = [1, 2, 3]
                    tile = src.read(RGB, window=window) # (c, h, w)

                    left, bottom, right, top = bounds(window, src.transform)
                    lon_left, lat_bot, lon_right, lat_top = transform_bounds(
                        src.crs, dst_crs, left, bottom, right, top
                    )

                    center_lon = (lon_left + lon_right) / 2
                    center_lat = (lat_bot + lat_top) / 2

                    metadata = {
                        "center_lon": center_lon,
                        "center_lat": center_lat,
                    }

                    yield tile, metadata

    except rasterio.RasterioIOError as e:
        raise RuntimeError(f"Something went wrong: {path}") from e


def flush(
    tile_batch: list[np.ndarray],
    metadata_batch: list[dict],
    model: BaseModel,
    writer: TileWriter,
) -> None:
    """Flush the batch to DB."""
    if not tile_batch or not metadata_batch:
        return

    if len(tile_batch) != len(metadata_batch):
        raise RuntimeError("Batch sizes do not match")
    
    embeddings = model.embed_batch(tile_batch)
    writer.write_batch(metadata_batch, embeddings)

    tile_batch.clear()
    metadata_batch.clear()


def embed_raster(
    file_path: str,
    model: BaseModel,
    writer: TileWriter,
    stride: int = 256,
    window_size: int = 512,
    batch_size: int = 32,
    dst_crs: str = "EPSG:4326",
) -> None:
    path = Path(file_path)

    if not path.exists():
        raise ValueError("The path does not exist")

    tile_batch = []
    metadata_batch = []

    for tile, metadata in iterate_raster(
        path,
        stride,
        window_size,
        dst_crs
    ):
        tile_batch.append(tile)
        metadata_batch.append(metadata)

        if len(tile_batch) == batch_size:
            flush(
                tile_batch=tile_batch,
                metadata_batch=metadata_batch,
                model=model,
                writer=writer,
            )

    flush(
        tile_batch=tile_batch,
        metadata_batch=metadata_batch,
        model=model,
        writer=writer,
    )
