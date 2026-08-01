import numpy as np

import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, hsv_to_rgb

from Algos.Create_Patterns import create_patterns, create_gaussians, create_gaussians_moving




def data_type(type):
    if type == '2 gaussian 2.25sig':
        dff1, params = create_gaussians(N=64, M=128, frames=65, num_gaus=2, x0=32, y0=70 - 18, sd0=16, t0_0=22 + 22.5,sdT0=10, x1=32, y1=70 + 18, sd1=16, t0_1=22, sdT1=10)
        title = r'2 Gaussians $\Delta$X=$\Delta$T=2.25$\sigma$'

        return dff1, title

    if type == 'radial':
        dff1 = create_patterns(N=128, M=64, frames=49, pattern='radial', x0=32, y0=80, sd0=2, u=0, v=1, rad_spd=0.5,rad_width=3)
        title = 'Radial Wave'

        return dff1, title


def schematic_figure():
    brain_mask = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/brain_mask.npy')
    brain_mask = brain_mask[:, :64]
    dff1, title = data_type('radial')
    dff2, title = data_type('2 gaussian 2.25sig')
    while brain_mask.ndim < dff1.ndim:
        brain_mask = np.expand_dims(brain_mask, axis=-1)
    dff1 *= brain_mask
    dff2 *= brain_mask

    index = 5
    frame_indices = [index + 1, index + 7, index + 13, index + 17, index + 21, index + 27, index + 33]  # , index + 36]

    # Load outer line
    outer_line_rgb = np.load(
        '/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/outer_line_rgb.npy')
    outer_line_rgb = np.where(outer_line_rgb[:, :, 0] == 1, np.nan, outer_line_rgb[:, :, 0])

    figsize_cm = (9, 4.5)  # example in cm
    figsize_in = tuple(x / 2.54 for x in figsize_cm)  # convert to inches

    fig = plt.figure(figsize=figsize_in, dpi=350)

    # 2 rows, len(frame_indices) + 1 columns (last one for colorbar)
    gs = gridspec.GridSpec(2, len(frame_indices) + 2,
                           width_ratios=[1] * len(frame_indices) + [0.3, 0.85],
                           height_ratios=[1, 1],
                           wspace=0.05, hspace=0.1)

    axes_row1 = [fig.add_subplot(gs[0, i]) for i in range(len(frame_indices))]
    axes_row2 = [fig.add_subplot(gs[1, i]) for i in range(len(frame_indices))]
    cax = fig.add_subplot(gs[0, -2])

    colors = [
        (1.0, 1.0, 1.0),  # white
        (0.5, 0.7, 0.8),  # pale blue
        (0.1, 0.2, 0.6),  # navy blue
    ]
    cmap = LinearSegmentedColormap.from_list("AbyssBlue", colors)

    colors = [(0.9375, 0.725, 0.725, 1.0), (0.7425, 0.275, 0.3, 1.0), (0.3875, 0.05, 0.115, 1.0)]

    for i, frame_idx in enumerate(frame_indices):
        axes_row1[i].imshow(dff1[:, :, frame_idx], cmap=cmap, vmin=0, vmax=1)
        axes_row1[i].imshow(outer_line_rgb[:, :64], cmap='gray', alpha=0.6)
        axes_row1[i].axis('off')
        if i in [2, 3, 4]:
            x_middle = dff1.shape[1] // 2  # Middle of the image width (axis=1)

            axes_row1[i].plot([x_middle, x_middle], [8, 122], color=(0.37, 0.1, 0.18), linestyle='--', linewidth=0.3)
            axes_row1[i].text(x_middle + 1, 127, f"T{i - 1}",  # adjust x/y position
                              fontsize=4.5, color=colors[i - 2], fontname='Arial', fontweight='bold',
                              va='top', ha='center', rotation=0, )

    for i, frame_idx in enumerate(frame_indices):
        axes_row2[i].imshow(dff2[:, :, frame_idx], cmap=cmap, vmin=0, vmax=1)
        axes_row2[i].imshow(outer_line_rgb[:, :64], cmap='gray', alpha=0.6)
        axes_row2[i].axis('off')
        if i in [2, 3, 4]:
            x_middle = dff1.shape[1] // 2  # Middle of the image width (axis=1)

            axes_row2[i].plot([x_middle, x_middle], [8, 122], color=(0.37, 0.1, 0.18), linestyle='--', linewidth=0.3)
            axes_row2[i].text(x_middle + 1, 127, f"T{i - 1}",  # adjust x/y position
                              fontsize=4.5, color=colors[i - 2], fontname='Arial', fontweight='bold',
                              va='top', ha='center', rotation=0)
    # Add colorbar
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label("Amp", fontsize=4, rotation=0)
    cbar.ax.xaxis.set_label_coords(0.5, -0.9)  # x>1 moves right outside, y=0.5 center vertically
    cbar.set_ticks([])
    cbar.ax.tick_params(labelsize=5, width=0.4)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.45)  # thinner border
        # Or use spine.set_visible(False) to hide them completely

    pos = cax.get_position()
    cax.set_position([pos.x0 - 0.086, pos.y0 + 0.41, pos.width * 2.8, pos.height - 0.4])

    def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    temporal_ax_2, temporal_ax_1 = fig.add_subplot(gs[0, -1]), fig.add_subplot(gs[1, -1])  # takes both rows

    sigma = 1.1
    mu1 = -0.6 - 1.6 * sigma  # First Gaussian center
    mu2 = -1.25 + 1.6 * sigma  # Second Gaussian center

    x = np.linspace(-5, 5, 200)  # Create x axis

    y1 = 2.3 * gaussian(x, mu1, sigma)
    y2 = 2.3 * gaussian(x, mu2, sigma)

    time_factors = [[50 / 50, 25 / 50], [40 / 50, 40 / 50], [25 / 50, 50 / 50]]  # From rise to fall

    original_cmap = LinearSegmentedColormap.from_list("WineGradient", [
        (1.0, 1.0, 1.0),  # white
        (0.9, 0.5, 0.5),  # rose
        (0.25, 0.0, 0.1),  # deep wine
    ])

    start = 0
    end = 1

    subset_cmap = LinearSegmentedColormap.from_list("subset_cmap", original_cmap(np.linspace(start, end, 256)))
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=subset_cmap)
    sm.set_array([])

    cax2 = fig.add_subplot(gs[0, -1])

    cbar = fig.colorbar(sm, cax=cax2, orientation='horizontal')
    cbar.set_label("Time", fontsize=4, rotation=0)
    cbar.ax.xaxis.set_label_coords(0.5, -0.9)  # x>1 moves right outside, y=0.5 center vertically
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.ax.tick_params(labelsize=5, width=0.4)

    tick_positions = [0.3, 0.55, 0.8]
    tick_labels = ['T1', 'T2', 'T3']

    colors = {
        "T1": (0.9375, 0.725, 0.725, 1.0),
        "T2": (0.7425, 0.275, 0.3, 1.0),
        "T3": (0.3875, 0.05, 0.115, 1.0)
    }
    for pos, label in zip(tick_positions, tick_labels):
        color = subset_cmap(pos)
        # Use transformation to place text in data coords (x) and axis fraction (y)
        cbar.ax.text(
            pos, 1.85,  # x in data coords, y in axis fraction coords (just below -1 moves text down)
            label,
            ha='center', va='center', fontweight='bold',
            fontsize=3.5,
            color=colors[label],
            transform=cbar.ax.transData  # x as data (0–1), y as relative axis (0=bottom, 1=top)
        )
        cbar.ax.plot(
            [pos, pos], [1, 1.3],  # start and end of the tick (in axis y-coords)
            transform=cbar.ax.transData,
            color=color,
            linewidth=0.5,
            clip_on=False
        )
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.45)  # thinner border

    pos = cax2.get_position()
    cax2.set_position([pos.x0, pos.y0 + 0.41, pos.width, pos.height - 0.4])

    times = [subset_cmap(0.25), subset_cmap(0.55), subset_cmap(0.85)]
    for i, t in enumerate(time_factors):
        temporal_ax_1.plot(y1 * t[0] + y2 * t[1], x, color=times[i], linewidth=0.7)

    # Simulate the rise and fall from y=0 to y=1 and back down
    y_shifted_1 = 2.4 * np.exp(-0.5 * (x - 0.2) ** 2) / np.sqrt(2 * np.pi)
    y_shifted_3 = 2.4 * np.exp(-0.5 * (x + 0.8) ** 2) / np.sqrt(2 * np.pi)
    y_shifted_5 = 2.4 * np.exp(-0.5 * (x + 1.8) ** 2) / np.sqrt(2 * np.pi)

    temporal_ax_2.plot(y_shifted_5, x, label='y_shifted_5', color=times[0], linewidth=0.7)
    temporal_ax_2.plot(y_shifted_3, x, label='y_shifted_3', color=times[1], linewidth=0.7)
    temporal_ax_2.plot(y_shifted_1, x, label='y_shifted_1', color=times[2], linewidth=0.7)

    temporal_ax_1.set_xticks([])
    temporal_ax_1.set_yticks([])
    temporal_ax_2.set_xticks([])
    temporal_ax_2.set_yticks([])

    for spine in temporal_ax_1.spines.values():
        spine.set_linewidth(0.5)  # or smaller value, e.g., 0.3 or 0.2
    for spine in temporal_ax_2.spines.values():
        spine.set_linewidth(0.5)  # or smaller value, e.g., 0.3 or 0.2

    temporal_ax_1.spines['right'].set_visible(False)
    temporal_ax_1.spines['top'].set_visible(False)
    temporal_ax_2.spines['right'].set_visible(False)
    temporal_ax_2.spines['top'].set_visible(False)

    fig.text(0.11, 0.69, 'Travelling Wave', va='center', ha='center',
             rotation='vertical', fontsize=5, fontname='Arial')

    fig.text(0.11, 0.29, 'Modular Sequence', va='center', ha='center',
             rotation='vertical', fontsize=5, fontname='Arial')

    fig.text(0.08, 0.89, 'a', va='center', ha='center',
             fontsize=6, fontname='Arial')

    fig.text(0.08, 0.49, 'b', va='center', ha='center',
             fontsize=6, fontname='Arial')

    # plt.tight_layout()
    plt.savefig("cortex_double_row.pdf", bbox_inches='tight', dpi=1000)
    plt.show()


schematic_figure()