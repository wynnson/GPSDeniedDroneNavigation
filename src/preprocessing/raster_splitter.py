import torch
import torch.nn.functional as F
import rasterio
import numpy as np

from torch import nn
from collections.abc import Iterable
from rasterio.windows import Window, bounds
from rasterio.warp import transform_bounds
from pathlib import Path

from src.preprocessing.writer import TileWriter


def iterate_raster(
    path: Path,
    window_size: int,
    dst_crs: str = "EPSG:4326"
) -> Iterable[tuple[np.ndarray, dict]]:
    """
    Slides window through original raster, chopping it up
    into tiny pieces to embed.
    """
    try:
        with rasterio.open(path) as src:
            for r in range(0, src.height, window_size):
                for c in range(0, src.width, window_size):
                    width = min(window_size, src.width - c)
                    height = min(window_size, src.height - r)
                    window = Window(c, r, width, height)

                    RGB = [1, 2, 3]
                    tile = src.read(RGB, window=window) # (c, h, w)

                    pad_h = window_size - tile.shape[1]
                    pad_w = window_size - tile.shape[2]

                    if pad_h > 0 or pad_w > 0:
                        tile = np.pad(
                            tile, 
                            pad_width=((0,0), (0, pad_h), (0, pad_w)), 
                            mode="constant",
                            constant_values=0
                        )

                    left, bottom, right, top = bounds(window, src.transform)
                    lon_left, lat_bot, lon_right, lat_top = transform_bounds(
                        src.crs, dst_crs, left, bottom, right, top
                    )

                    metadata = {
                        "lon_left": lon_left,
                        "lat_bottom": lat_bot,
                        "lon_right": lon_right,
                        "lat_top": lat_top
                    }

                    yield tile, metadata

    except rasterio.RasterioIOError as e:
        raise RuntimeError(f"Something went wrong: {path}") from e


def preprocess_tile_batch_dinov2(tile_batch, device) -> torch.Tensor:
    """Normalizes batch tiles to what dinov2 expects."""
    size = (518, 518) # DINO v2 native

    # Imagenet defaults
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    batch = torch.from_numpy(np.stack(tile_batch)).float().to(device) / 255.0
    batch = F.interpolate(
        batch,
        size=size,
        mode="bilinear",
        align_corners=False
    )

    return (batch - mean) / std


def flush(
    tile_batch: list[np.ndarray],
    metadata_batch: list[dict],
    model: nn.Module,
    writer: TileWriter,
    device: str
) -> None:
    """Flush the batch to DB."""
    if not tile_batch or not metadata_batch:
        return

    if len(tile_batch) != len(metadata_batch):
        raise RuntimeError("Batch sizes do not match")
    
    normalized = preprocess_tile_batch_dinov2(tile_batch, device=device)

    with torch.inference_mode():
        embeddings = model(normalized)

    embeddings = embeddings.cpu().numpy()
    writer.write_batch(metadata_batch, embeddings)

    tile_batch.clear()
    metadata_batch.clear()
    

def embed_raster(
    file_path: str,
    model: nn.Module,
    writer: TileWriter,
    window_size: int = 512,
    batch_size: int = 32,
    dst_crs: str = "EPSG:4326",
    device: str ="cpu",
) -> None:
    path = Path(file_path)

    if not path.exists():
        raise ValueError("The path does not exist")

    tile_batch = []
    metadata_batch = []

    for tile, metadata in iterate_raster(path, window_size, dst_crs):
        tile_batch.append(tile)
        metadata_batch.append(metadata)

        if len(tile_batch) == batch_size:
            flush(
                tile_batch=tile_batch,
                metadata_batch=metadata_batch,
                model=model,
                writer=writer,
                device=device
            )

    flush(
        tile_batch=tile_batch,
        metadata_batch=metadata_batch,
        model=model,
        writer=writer,
        device=device
    )
