import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from math import ceil
from rasterio.windows import Window
from pathlib import Path


RGB = [1, 2, 3]


def show_all_tiles(
    path: Path,
    stride: int,
    window_size: int,
):
    """Visualizes all tiles in a tif"""
    with rasterio.open(path) as src:
        image = src.read(RGB)   # C, H, W
        image = np.moveaxis(image, 0, -1)   # H, W, C
        fig, ax = plt.subplots(figsize=(25,25))
        ax.imshow(image)

        tile_id = 0
        for r in range(0, src.height, stride):
            for c in range(0, src.width, stride):
                width = min(window_size, src.width - c)
                height = min(window_size, src.height - r)

                if width < window_size or height < window_size:
                    continue

                center_x = c + width / 2
                center_y = r + height / 2

                rect = patches.Rectangle(
                    (c, r),
                    width,
                    height,
                    edgecolor="black",
                    alpha=0.9,
                    linewidth=1.2,
                    zorder=1,
                    fill=False
                )
                ax.add_patch(rect)

                ax.scatter(
                    center_x,
                    center_y,
                    s=50,
                    marker="o",
                    color="cyan",
                    alpha=0.9,
                    zorder=10
                )

                ax.text(
                    center_x,
                    center_y,
                    str(tile_id),
                    fontsize=14,
                    ha="center",
                    va="center",
                    color="white",
                    zorder=50
                )

                tile_id += 1

        ax.axis("off")
        plt.tight_layout()
        plt.show()


def show_predictions(
    path: Path,
    predictions: list[tuple],
    stride: int,
    window_size: int = 512
):
    with rasterio.open(path) as src:
        tiles_per_row = ((src.width - window_size) // stride) + 1

        num_predictions = len(predictions)

        max_cols = 5
        ncols = min(num_predictions, max_cols)
        nrows = ceil(num_predictions / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
        axes = np.atleast_1d(axes).flatten()

        top = 1
        for ax, prediction in zip(axes, predictions):
            tile_id, score, coords = prediction

            row = tile_id // tiles_per_row
            col = tile_id % tiles_per_row

            r = row * stride
            c = col * stride

            window = Window(c, r, window_size, window_size)
            image = src.read(RGB, window=window)

            image = np.moveaxis(image, 0, -1)
            lon, lat = coords

            ax.imshow(image)

            ax.set_title(
                f"Tile {tile_id}\n"
                f"Top {top}\n"
                f"Score: {score:.3f}\n"
                f"{lat:.5f}, {lon:.5f}",
                fontsize=7
            )

            ax.axis("off")
            top += 1

        plt.tight_layout()
        plt.show()