from PIL import Image
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import gc
import requests
import numpy as np
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


def visualize_sample(input_curves, box_lim):
    batch_size = len(input_curves)
    dpi = 100
    figure_size_inches = ( 224 / dpi, 224 / dpi)
    out = []
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    
    for in_curve in input_curves:

        # import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        fig.set_size_inches(figure_size_inches)
        # fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

        draw_curves(in_curve, ax=ax, box_lim=box_lim, color="black")
        # draw_curves(label_curves[i], ax=ax, box_lim=box_lim, color="blue")

        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        # plt.pause(3.)
        plt.close()
        # plt.cla()
        # plt.clf()
        # plt.close('all')
        # plt.close(fig)
        # gc.collect()
        
        del fig, ax
        out.append(img)
        
        # img.close()
        # image = Image.open(requests.get(url, stream=True).raw)
        # out.append(image)

    # gc.collect()
    # del matplotlib, plt
    return out


def visualize_sample_cv(point_entities, box_lim):
    dpi = 100
    figure_size_inches = ( 224 / dpi, 224 / dpi)
    out = []
    
    for entities in point_entities:
        np_image = render_sketch_opencv(entities, size=224, quantize_bins=64)
        pil_image = np_image[:, :, ::-1]  # BGR to RGB
        img = Image.fromarray(pil_image, mode='RGB')
        out.append(img)

    return out

def visualize_sample_pil(point_entities, box_lim):
    out = []
    
    for entities in point_entities:
        pil_image = render_sketch_pil(entities, figure_size_pixels=224, pad_in_pixels=10, linewidth=2)
        out.append(pil_image)

    return out

def draw_curves(curves, ax, box_lim, color, draw_points=False):
    ax.set_xlim(left=-3, right=box_lim)
    ax.set_ylim(bottom=-3, top=box_lim)
    ax.set_xticks([])
    ax.set_yticks([])

    colors = {2: 'red', 3:'green', 4:'blue'}

    for curve in curves:
        if curve and curve.good:
                curve.draw(ax=ax,  color=colors[curve.points.shape[0]], draw_points=draw_points)
            

def render_sketch_opencv(point_entities, size, quantize_bins, linewidth=2):

    np_image = np.ones((size, size, 3), np.uint8) * 255
    cell_size = size // quantize_bins

    curves = get_curves(point_entities)
    assert all(curve for curve in curves)

    for curve in curves:
        curve.draw_np(np_image, draw_points=True, linewidth=linewidth, cell_size=cell_size)

    return np_image