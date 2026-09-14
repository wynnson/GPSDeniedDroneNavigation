import argparse
import time
import torch
import yaml

from pathlib import Path

from src.preprocessing.raster_splitter import embed_raster
from src.preprocessing.writer import TileWriter
from src.utils.device import get_device


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args(config):
    parser = argparse.ArgumentParser(
        description="Embed a georeferenced raster into FAISS + SQLite"
    )

    parser.add_argument(
        "--raster",
        type=Path,
        default=Path("data/GT_NW.tif"),
        help="Path to input GeoTIFF"
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=512
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/tiles.db")
    )

    parser.add_argument(
        "--faiss",
        type=Path,
        default=Path("data/tiles.faiss")
    )

    parser.add_argument(
        "--device",
        default=None
    )

    return parser.parse_args()


def main():
    config = load_config(Path("src/config/default.yaml"))
    args = parse_args(config)
    device = args.device or get_device()

    print(f"Device: {device}")

    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14"
    )

    writer = TileWriter(
        db_path=args.db,
        faiss_path=args.faiss,
        embedding_dim=384
    )

    start = time.perf_counter()

    try:
        embed_raster(
            file_path=str(args.raster),
            model=model,
            writer=writer,
            window_size=args.window_size,
            batch_size=args.batch_size,
            device=device
        )

        elapsed_time = time.perf_counter() - start
        print(f"Successful! Finished in {elapsed_time:.2f} seconds")

    finally:
        writer.close()


if __name__ == "__main__":
    main()    
