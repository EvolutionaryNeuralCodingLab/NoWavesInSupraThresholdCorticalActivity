import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap


def cyclic_hsv_cmap(n=256):
    hue = np.linspace(0, 1, n + 1)
    # Your custom variations:
    saturation = 0.7 + 0.3 * np.sin(2 * np.pi * hue)
    value = 0.85 + 0.15 * np.cos(2 * np.pi * hue)
    hsv = np.stack([hue, saturation, value], axis=1)
    rgb = hsv_to_rgb(hsv.reshape(1, -1, 3)).squeeze()
    rgb = rgb[:-1]
    return LinearSegmentedColormap.from_list("cyclic_hsv", rgb)


# 1. Generate your custom colormap
cmap = cyclic_hsv_cmap()
cmap.set_bad(color='white')


def plot_circular_colormap(cmap, n_theta=512, n_r=100):
    theta_edges = np.linspace(0, 2 * np.pi, n_theta + 1)
    r_edges = np.linspace(0, 1, n_r + 1)

    theta, r = np.meshgrid(theta_edges, r_edges)

    # Cell centers, because pcolormesh colors are per cell
    theta_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])
    theta_c, r_c = np.meshgrid(theta_c, r_c)

    hue_values = (theta_c / (2 * np.pi)) % 1

    # Get original cyclic colors
    colors = cmap(hue_values)

    # Fade toward white in the center
    # Larger exponent = whiter center extends more outward
    fade = r_c[..., None] ** 0.5

    colors[..., :3] = 1 - fade * (1 - colors[..., :3])

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(3, 3))

    mesh = ax.pcolormesh(theta, r, np.zeros_like(theta_c), shading='flat')
    mesh.set_array(None)
    mesh.set_facecolors(colors.reshape(-1, 4))

    angle_positions = {
        0: r"$\frac{\pi}{2}$",
        np.pi / 2: r"$0$",
        np.pi: r"$\frac{3\pi}{2}$  ",
        3 * np.pi / 2: r"$\pi$"
    }

    for angle, label in angle_positions.items():
        ax.text(angle, 1.4, label,
                ha='center', va='center',
                fontsize=78, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor('white')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(
        'circular_colormap.pdf',
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.5,
        facecolor='white'
    )
    plt.show()
# 4. Pass the custom colormap into the function
plot_circular_colormap(cmap)