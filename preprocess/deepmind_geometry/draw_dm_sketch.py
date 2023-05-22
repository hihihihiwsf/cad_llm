import matplotlib.pyplot as plt


def draw_sketch(sketch):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    for entity in sketch:
        if not entity:
            print("none entity")
            continue
        if entity.exception:
            print(entity.exception)
            continue

        entity.draw(ax)

    plt.close(fig)
