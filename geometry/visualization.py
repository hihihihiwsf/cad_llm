import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from geometry.parse import get_curves


def visualize_batch(input_curves, label_curves, sample_curves, box_lim):
    batch_size = len(input_curves)

    dpi = 100
    figure_size_inches = (2 * 512 / dpi, batch_size * 512 / dpi)

    fig, axes = plt.subplots(batch_size, 2)
    fig.set_dpi(dpi)
    fig.set_size_inches(figure_size_inches)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    for i in range(batch_size):
        ax = axes[i, 0]
        ax.set_title("Prompt + Ground Truth")
        draw_curves(input_curves[i], ax=ax, box_lim=box_lim, color="black")
        draw_curves(label_curves[i], ax=ax, box_lim=box_lim, color="blue")

        ax = axes[i, 1]
        ax.set_title("Prompt + Sample")
        draw_curves(input_curves[i], ax=ax, box_lim=box_lim, color="black")
        draw_curves(sample_curves[i], ax=ax, box_lim=box_lim, color="red")

    plt.close()
    return fig


def draw_curves(curves, ax, box_lim, color, draw_points=False):
    ax.set_xlim(left=-box_lim, right=box_lim)
    ax.set_ylim(bottom=-box_lim, top=box_lim)
    ax.set_xticks([])
    ax.set_yticks([])

    for curve in curves:
        if curve and curve.good:
            curve.draw(ax=ax,  color=color, draw_points=draw_points)


def render_sketch_opencv(point_entities, size, quantize_bins, linewidth=2):
    np_image = np.ones((size, size, 3), np.uint8) * 255
    cell_size = size // quantize_bins

    curves = get_curves(point_entities)
    assert all(curve for curve in curves)

    for curve in curves:
        curve.draw_np(np_image, draw_points=True, linewidth=linewidth, cell_size=cell_size)

    return np_image


def render_sketch_pil(point_entities, figure_size_pixels, pad_in_pixels, min_val=0, max_val=63, linewidth=2):
    """
    Raises exception on bad sketches
    """
    img = Image.new("RGB", size=(figure_size_pixels, figure_size_pixels), color=(255, 255, 255))
    img_draw = ImageDraw.Draw(img)

    curves = get_curves(point_entities)
    assert all(curve for curve in curves)

    scale_by = figure_size_pixels - pad_in_pixels
    shift_by = 0.5 * figure_size_pixels
    val_range = max_val + 1 - min_val

    def rescale(x):
        x = (x / val_range) - 0.5  # normalize between -0.5 and 0.5
        x = scale_by * x + shift_by  # scale to canvas
        return x

    for curve in curves:
        curve.draw_pil(img_draw, draw_points=True, linewidth=linewidth, transform=rescale)

    # Origin is at top-left in image space, so flip
    img = img.transpose(method=Image.FLIP_TOP_BOTTOM)

    return img
