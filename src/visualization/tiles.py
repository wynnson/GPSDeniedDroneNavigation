import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from math import ceil
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from pathlib import Path


RGB = [1, 2, 3]


def show_all_tiles(
    path: Path,
    stride: int,
    window_size: int,
    predictions: list[tuple] = None
):
    """Visualizes all tiles in a tif"""
    with rasterio.open(path) as src:
        image = src.read(RGB)   # C, H, W
        image = np.moveaxis(image, 0, -1)   # H, W, C
        fig, ax = plt.subplots(figsize=(25,25))
        ax.imshow(image)

        prediction_map = {}
        if predictions is not None:
            for prediction in predictions:
                uid, score, _ = prediction
                prediction_map[uid] = score

        tile_id = 0
        for r in range(0, src.height, stride):
            for c in range(0, src.width, stride):
                width = min(window_size, src.width - c)
                height = min(window_size, src.height - r)

                if width < window_size or height < window_size:
                    continue

                center_x = c + width / 2
                center_y = r + height / 2

                if tile_id in prediction_map:
                    rect_fill = True
                    rect_facecolor = "red"
                    rect_alpha = prediction_map[uid] ** 5 # score
                    scatter_marker = "d"
                    scatter_color = "orange"
                    s = 500
                else:
                    rect_fill = False
                    rect_facecolor = None
                    rect_alpha = .9
                    scatter_marker = "o"
                    scatter_color = "cyan"
                    s = 50

                rect = patches.Rectangle(
                    (c, r),
                    width,
                    height,
                    edgecolor="black",
                    alpha=rect_alpha,
                    linewidth=1.2,
                    facecolor=rect_facecolor,
                    zorder=1,
                    fill=rect_fill
                )

                ax.add_patch(rect)

                ax.scatter(
                    center_x,
                    center_y,
                    s=s,
                    marker=scatter_marker,
                    color=scatter_color,
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
    """Visualizes the top k tiles in prediction"""
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


def show_estimated_position(
    path: Path,
    estimated_position: tuple[float, float], 
    predictions: list[tuple],
    score_epsilon: float,
    dst_crs: str = "EPSG:4326",
):
    est_lon, est_lat = estimated_position

    with rasterio.open(path) as src:
        image = src.read(RGB)
        image = np.moveaxis(image, 0, -1)
        left, bottom, right, top = transform_bounds(
            src.crs,
            dst_crs,
            *src.bounds
        )

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image, extent=[left, right, bottom, top], origin="upper")

    for uid, score, (lon, lat) in predictions:
        if score < score_epsilon:
            continue

        ax.scatter(lon, lat, s=60, alpha=score, zorder=2, marker="d")
        ax.plot(
            [lon, est_lon],
            [lat, est_lat],
            alpha=score ** 4,
            linewidth=1.5,
            zorder=1
        )
        ax.text(lon, lat, str(uid), fontsize=8)

    ax.scatter(
        est_lon, est_lat, s=180, marker="^", zorder=3, label="Estimated Position", color="cyan"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.show()
