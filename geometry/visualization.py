import matplotlib.pyplot as plt
from PIL import Image

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
    figure_size_inches = ( 512 / dpi, 512 / dpi)
    out = []
    for in_curve in input_curves:


        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        fig.set_size_inches(figure_size_inches)
        # fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

        draw_curves(in_curve, ax=ax, box_lim=box_lim, color="black")
        # draw_curves(label_curves[i], ax=ax, box_lim=box_lim, color="blue")

        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        plt.close()
        out.append(img)

    return out


def draw_curves(curves, ax, box_lim, color, draw_points=False):
    ax.set_xlim(left=0, right=box_lim)
    ax.set_ylim(bottom=0, top=box_lim)
    ax.set_xticks([])
    ax.set_yticks([])

    colors = {2: 'red', 3:'green', 4:'blue'}

    for curve in curves:
        if curve and curve.good:
            curve.draw(ax=ax,  color=colors[curve.points.shape[0]], draw_points=draw_points)
            