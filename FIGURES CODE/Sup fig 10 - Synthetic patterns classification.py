import numpy as np
from scipy.signal import hilbert
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, hsv_to_rgb

from Algos.Display import plot_quiver
from Algos.Create_Patterns import create_patterns, create_gaussians, create_gaussians_moving
from Algos.Data_Processing import Filter, resize, decrease_frame_rate, normalize_data
from Algos.Horn_Schunck import horn_schunck , horn_schunck_phase

#### Algorithm parameters###
alpha = 0.5 ## Optic Flow Horn-Schunck alpha parameter
iterations = 500  ## Number of iterations
n = 3  ## neighborhood size (2n+1)x(2n+1) around each pixel
lim = 0.5  ## Threshold of Wavness values for final score
beta = 0.035

theta = 15  ## Theta threshold for uniform continuity

plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42


class PreDataProcessing:
    def __init__(self, data):
        self.N = data.shape[0]
        self.M = data.shape[1]
        self.frames = data.shape[2]
        self.dff = data

    def resize(self, m, n):
        resized_dff = np.zeros((m, n, self.frames))
        for i in range(self.frames):
            resized_dff[:, :, i] = resize(self.dff[:, :, i], m, n)
        self.dff = resized_dff
        self.N = resized_dff.shape[0]
        self.M = resized_dff.shape[1]

    def filter(self, fil, sigma, kernel):
        ## filter = 'gaussian' , 'bilateral' , 'average' , 'normalize'
        filtered_dff = np.zeros((self.dff.shape[0], self.dff.shape[1], self.dff.shape[2]))
        for i in range(self.dff.shape[2]):
            filtered_dff[:, :, i] = Filter(self.dff[:, :, i], fil=str(fil), sigma=sigma, kernel=kernel)
        self.dff = filtered_dff

    def add_noise(self, std):
        for t in range(self.frames):
            self.dff[:, :, t] += np.random.normal(0, std, size=(self.N, self.M))

    def add_noise_temporal(self, std):
        for i in range(self.N):
            for j in range(self.M):
                self.dff[i, j, :] += np.random.normal(0, std, size=self.frames)

    def delta_f_over_f(self, T1, T2):
        """
        Compute ΔF/F for a video with a floating minimum baseline.

        Parameters:
        - video: 3D numpy array of shape (N, M, frames) representing the video.
        - T1: Integer representing the size of the sliding window for the average filter.
        - T2: Integer representing the size of the sliding window for the minimum filter among the averages.

        Returns:
        - delta_f_over_f_video: 3D numpy array of the same shape as the input video, containing the ΔF/F values.

        Based on 'In vivo two-photon imaging of sensory-evoked dendritic calcium signal in cortal neurons" - Arthur Konnerth
        """

        delta_f_over_f_video = np.zeros_like(self.dff, dtype=np.float64)

        # Iterate over each pixel position
        for i in range(self.dff.shape[0]):
            for j in range(self.dff.shape[1]):
                window = np.ones(T1) / T1

                # Apply convolution to compute the rolling average
                floating_avg = np.convolve(self.dff[i, j, :], window, mode='same')

                # Compute the floating minimum using a minimum filter
                baseline_f = minimum_filter(floating_avg, size=T2)

                delta_f = self.dff[i, j, :] - baseline_f

                delta_f_over_f = delta_f / baseline_f

                # Explicitly set ΔF/F to zero where baseline was zero
                zero_mask = (baseline_f == 0)
                delta_f_over_f[zero_mask] = 0  # Set ΔF/F to zero where baseline is zero

                # Store the result in the output array
                delta_f_over_f_video[i, j, :] = delta_f_over_f

        self.dff = delta_f_over_f_video

class FlowAnalyze:
    def __init__(self, data):
        self.dff = data.dff
        self.N = data.dff.shape[0]
        self.M = data.dff.shape[1]
        self.frames = data.dff.shape[2]

        self.n = n
        self.theta = theta
        self.iter = iterations

        self.flows = []
        self.converge = 0

        self.phase_space = np.zeros((self.N, self.M, self.frames - 1, 2))
        self.abs_phase_space = np.zeros((self.N, self.M, self.frames - 1, 2))
        self.velocities = np.zeros((self.N, self.M, self.frames - 1, 2))
        self.space_and_time = np.zeros((self.N, self.M, self.frames - 1, 3))

        self.sum_phase_space = np.zeros((self.N, self.M, self.frames - 1, 2))
        self.abs_sum_phase_space = np.zeros((self.N, self.M, self.frames - 1, 2))

        self.waveness = np.zeros((self.N, self.M, 4))  ### 4 dimensions are for ֿ-> (R,G,B,SCORE)
        self.mask = np.zeros((self.N, self.M))
        self.Flow = True

    def horn_schunck_flow(self, alpha, num_iter,phase):
        all_convergence = []

        for i in range(self.frames - 1):
            image1 = self.dff[:, :, i]
            image2 = self.dff[:, :, i + 1]

            if phase == True:
                flow, convergence = horn_schunck_phase(image1, image2, alpha, num_iter)
            else:
                flow, convergence = horn_schunck(image1, image2, alpha, num_iter)

            #### If phase map

            #phase1 = np.mod(self.dff[:, :, i], 2 * np.pi)
            #phase2 = np.mod(self.dff[:, :, i + 1], 2 * np.pi)
            #image1 = np.exp(1j * phase1)
            #image2 = np.exp(1j * phase2)
            #flow, convergence = horn_schunck(image1, image2, alpha, num_iter)

            self.flows.append(flow)

            self.velocities[:, :, i, 0] = flow[:, :, 0]
            self.velocities[:, :, i, 1] = flow[:, :, 1]

            self.space_and_time[:, :, i, 0] = self.dff[:, :, i]
            self.space_and_time[:, :, i, 1] = flow[:, :, 0]
            self.space_and_time[:, :, i, 2] = flow[:, :, 1]

            self.phase_space[:, :, i, 0] = flow[:, :, 0] #* self.dff[:, :, i]
            self.phase_space[:, :, i, 1] = flow[:, :, 1] #* self.dff[:, :, i]

            all_convergence.append(convergence)
        self.converge = all_convergence

        def velocities():
            # Convert to NumPy array of shape (num_curves, num_iterations)
            all_convergence_array = np.array(all_convergence)

            # Compute mean and std across curves
            avg_convergence = np.mean(all_convergence_array, axis=0)
            std_convergence = np.std(all_convergence_array, axis=0)

            # Plot mean with shaded std band
            iterations = range(1, len(avg_convergence) + 1)

            plt.plot(iterations, avg_convergence, marker='.', color='teal', label='Mean Convergence')
            plt.fill_between(iterations,
                             avg_convergence - std_convergence,
                             avg_convergence + std_convergence,
                             color='teal', alpha=0.3, label='±1 Std Dev')

            plt.xlabel('Iteration')
            plt.ylabel('Difference in Magnitude ')
            plt.title('Horn-Schunck Convergence with Std Deviation - Cortex III')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        #means = np.mean(np.array(all_convergence), axis=0)
        #plt.figure(figsize=(8, 5))
        #plt.plot(means, linewidth=2)
        #plt.xlabel("Iteration")
        #plt.ylabel("Mean flow update (L2 norm)")
        #plt.title("Horn–Schunck Convergence")
        #plt.grid(True, alpha=0.3)
        #plt.xlim([0,150])
        #plt.tight_layout()
        #plt.show()

class Display:
    def __init__(self, data):
        self.dff = data.dff
        self.N = data.dff.shape[0]
        self.M = data.dff.shape[1]
        self.frames = data.dff.shape[2]
        self.data = data
        try:
            if data.Flow:
                self.flows = data.flows
                self.phase_space = data.phase_space
            if data.Waveness:
                self.sum_phase_space = data.sum_phase_space
                self.mask = data.mask
        except AttributeError:
            pass

        # original_blues = cm.get_cmap('Blues')
        # new_colors = np.vstack((np.array([1, 1, 1, 1]), original_blues(np.linspace(0, 1, 256))))  # Create a new colormap that starts with white and then includes the original Blues colors
        # self.color_map = ListedColormap(new_colors, name='WhiteBlues')  # Assign to instance attribute

        colors = [
            (1.0, 1.0, 1.0),  # white
            (0.5, 0.7, 0.8),  # pale blue
            (0.1, 0.2, 0.6),  # navy blue
        ]

        # Custom positions: move pale blue and navy blue earlier
        positions = [0.0, 0.5, 1]  # white at 0, pale blue at 0.2, navy at 0.5 (clipped early)

        self.color_map = LinearSegmentedColormap.from_list("AbyssBlue", list(zip(positions, colors)))

        self.n = n
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.iter = iterations
        self.lim = lim

    def plot_data(self):
        def cyclic_hsv_cmap(n=256):
            hue = np.linspace(0, 1, n + 1)
            saturation = 0.7 + 0.3 * np.sin(2 * np.pi * hue)
            value = 0.85 + 0.15 * np.cos(2 * np.pi * hue)
            hsv = np.stack([hue, saturation, value], axis=1)
            rgb = hsv_to_rgb(hsv.reshape(1, -1, 3)).squeeze()
            rgb = rgb[:-1]
            return LinearSegmentedColormap.from_list("cyclic_hsv", rgb)
        fig, ax = plt.subplots()
        # title = "test"
        # self.title="sda"
        brain_mask = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/brain_mask_64.npy')
        brain_mask = brain_mask[:, :]
        outer_line_rgb = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/outer_line_rgb_64.npy')

        def phase_cmap():
            smooth_cycle = mcolors.LinearSegmentedColormap.from_list(
                "smooth_cycle",
                [
                    "#d45568",
                    "#9d4edd",
                    "#6c6fbf",
                    "#3485a8",
                    "#2a9d8f",
                    "#8ab17d",
                    "#e9c46a",
                    "#f4a261",
                    "#e76f51",
                    "#d45568",  # bridge
                ],
                N=1024
            )

            return smooth_cycle


        def anima(i):
            ax.cla()
            #ax.imshow(self.dff[:, :, i]*brain_mask[::-1,:32], cmap=phase_cmap(), vmin=-np.pi, vmax=np.pi,origin="lower")
            ax.imshow(self.dff[:, :, i]*brain_mask[::-1,:32], cmap=self.color_map, vmin=0, vmax=1,origin="lower")

            #ax.imshow(outer_line_rgba[:,:64])
            layout = outer_line_rgb[::-1, :32].astype(float)

            # If values are 0-255, normalize to 0-1
            if layout.max() > 1:
                layout = layout / 255.0

            # alpha = 1 where layout is 1, alpha = 0 elsewhere
            alpha_mask = (layout == 1).astype(float)

            # black RGB image
            black_layout = np.zeros((*layout.shape, 3))

            #ax.imshow(black_layout[:,:], alpha=alpha_mask)

            self.flows[i][:, :, 0][brain_mask[::-1,:32] == 0] = 0
            self.flows[i][:, :, 1][brain_mask[::-1,:32] == 0] = 0

            plot_quiver(ax, self.flows[i], spacing=6, scale=0.2, color='black')
            ax.set_ylim(self.dff.shape[0], 0)

            #ax.imshow(self.dff[:, :, i], cmap=phase_cmap(), vmin=-np.pi, vmax=np.pi)

            # Assume outer_line_rgb has shape (H, W, 3) and contains (0.392, 0.392, 0.392) where the line is

           # ax.set_title(fr"{self.N}x{self.M} {self.title} ")
            for spine in ax.spines.values():
                spine.set_visible(False)

            time_sec = i / 12.5
            ax.set_title(f"Time: {int(time_sec // 60)} min {int(time_sec % 60)} sec")
            #        transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white",
            #        bbox=dict(facecolor='black', alpha=0.5))

            # ax.axis('off')

            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])


            return ax

        ani = animation.FuncAnimation(fig, func=anima, frames=self.frames - 1, blit=False, interval=64)
        image1 = ax.imshow(self.dff[:, :, 0], cmap=self.color_map, vmin=0, vmax=1)
        # image1 = ax.imshow(self.dff[:,:,0], cmap=self.color_map, vmin=-0.01, vmax=0.01)
        #image1 = ax.imshow(self.dff[:,:,0], cmap=phase_cmap(), vmin=-np.pi, vmax=np.pi)

        cbar = plt.colorbar(image1, ax=ax)
        # cbar.set_ticks([0, 1])
        # cbar.set_ticklabels(['0', '1'])

        #plt.figure(figsize=(2.5, 2.5))  # bigger figure in inches
        #plt.imshow(data.dff[:, :64, 80]*1.2, self.color_map, vmin=0, vmax=1)  # keep pixelated look
        #plt.axis('off')  # remove axes
        #plt.savefig("Barrel Activation.png", dpi=300, bbox_inches='tight')
        #plt.show()

        #plt.close()

        ani.save(f"{self.N}x{self.M}x{self.frames} {self.title} .mp4", writer='ffmpeg', fps=12.5)

       # frame = self.dff[:, :, 123]

        # Create a figure with desired size
        #fig, ax = plt.subplots(figsize=(8, 8))  # size in inches, adjust as needed
        #ax.imshow(frame, cmap=self.color_map)
        #ax.axis('off')  # remove axes

        # Save with higher DPI
        #fig.savefig("Tim_murphy_cortex.png", dpi=300, bbox_inches='tight', pad_inches=0)
        #plt.close(fig)
        #plt.imsave("Tim_murphy_cortex.png", frame, cmap=self.color_map, dpi=300)

        plt.show()

    def full_analysis_3columns(self, space, scale, data_type):

        figsize_cm = (8, 4.5)
        figsize_in = tuple(x / 2.54 for x in figsize_cm)
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=figsize_in)  # 1 row, 4 columns
        data = self.data

        for axis in [ax, ax3]:
            axis.set_aspect('equal')

        outer_line_rgb = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/outer_line_rgb.npy')

        outer_line_rgba = np.ones((*outer_line_rgb.shape[:2], 4))  # Start with white and alpha = 1

        # Set the line color (dark gray) where needed
        line_mask = np.all(outer_line_rgb < 1.0, axis=2)  # Where the line is (i.e., not white)
        outer_line_rgba[..., :3][line_mask] = 0.392  # RGB to gray
        outer_line_rgba[..., 3] = 0.0  # Transparent everywhere
        outer_line_rgba[..., 3][line_mask] = 1.0  # Fully opaque where the line is

        self.sum_phase_space[self.sum_phase_space[:, :, -1, 0] == 0] = np.nan
        self.sum_phase_space[self.sum_phase_space[:, :, -1, 1] == 0] = np.nan

        if data_type == 'cortex':
            brain_mask = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/brain_mask.npy')
            outer_line_rgb = np.load(
                '/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/outer_line_rgb.npy')

            ax.imshow(outer_line_rgb)
            ax3.imshow(outer_line_rgb)
        ax.imshow(outer_line_rgba)

        plot_quiver(ax, self.sum_phase_space[:, :, -1, :], spacing=space, scale=scale, color='black', width=0.006)

        ax.set_ylim(self.dff.shape[0], 0)
        ax.set_xlim(0, self.dff.shape[1])
        ax.axis("off")



        flattened_values = data.waveness[:, :, 3][self.mask == 1]  ### for Retina
        flattened_values = flattened_values.flatten()

        ax2.hist(flattened_values, bins=20, range=(0, 1), color='tomato', alpha=0.8, rwidth=0.9, edgecolor='darkred')

        max_count = flattened_values.shape[0]

        ax2.set_yticks([0, max_count])  # Two ticks: 0 and the max count
        ax2.set_yticklabels([0, 1])  # Normalize the labels to 0 and 1 for clarity
        ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
        ax2.set_xlim(0, 1)  # x-axis range
        ax2.set_ylim(0, max_count)
        ax2.set_aspect(1.0 / ax2.get_data_ratio())  # Adjust aspect ratio based on data

        def ratio_over_lim(lim, flattened_values, mask_activity):
            total_above_lim = np.count_nonzero(flattened_values[flattened_values > lim])
            total_non_zero = np.count_nonzero(mask_activity)

            if total_non_zero > 0:
                ratio = total_above_lim / total_non_zero
            else:
                ratio = 0  # or handle the division by zero case as you prefer

            return ratio

        ratio = ratio_over_lim(lim, flattened_values, data.mask)

        print('Score : ', ratio)
        masked_values = np.where(data.mask == 1, data.waveness[:, :, 3], np.nan)

        # ax3.set_title(f'Data Map of Values > {self.lim} , n={self.n}')

        ax3.imshow(outer_line_rgba)
        norm = plt.Normalize(vmin=0, vmax=2 * np.pi)

        def cyclic_hsv_cmap(n=256):
            # Create n+1 points to ensure cyclic closure
            hue = np.linspace(0, 1, n + 1)

            # Example: full hue cycle (0 to 1)
            # You can tweak saturation and value to vary smoothly and cyclically
            saturation = 0.7 + 0.3 * np.sin(2 * np.pi * hue)
            value = 0.85 + 0.15 * np.cos(2 * np.pi * hue)

            hsv = np.stack([hue, saturation, value], axis=1)

            # Convert to RGB
            rgb = hsv_to_rgb(hsv.reshape(1, -1, 3)).squeeze()

            # Drop last color (same as first) to avoid duplicate
            rgb = rgb[:-1]

            return LinearSegmentedColormap.from_list("cyclic_hsv", rgb)

        cmap = cyclic_hsv_cmap()

        activity_mask = np.zeros_like(data.waveness)
        activity_mask[:, :, 3] = np.where((data.mask[:, :] == 1), 0.1, 0)
        activity_mask[:, :, :3] = 70 / 255  # gray tone
        ax3.imshow(activity_mask)

        data.waveness[:, :, 3] = np.where(data.waveness[:, :, 3] < self.lim, 0, data.waveness[:, :, 3])
        # np.save("sum_phase_space_54MRL_12to15__3121_3214.npy", self.sum_phase_space)

        ax3.imshow(data.waveness[:, :, 0], cmap=cmap, vmin=-np.pi, vmax=np.pi, alpha=data.waveness[:, :, 3])



        #data.waveness[:, :, 3] = np.where(data.waveness[:, :, 3] < lim, 0, data.waveness[:, :, 3])


        ax3.axis("off")
        cmap = plt.cm.hsv

        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 2 * np.pi))
        sm.set_array([])

        ax.set_aspect('equal')
        ax3.set_aspect('equal')

        fig2 = ax3.figure  # Get the figure that ax2 belongs to
        extent = ax3.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        # fig2.savefig("waveness_ax2_only_3121_3214.png", bbox_inches=extent, dpi=300, pad_inches=0)

        #plt.tight_layout()
        plt.savefig(f'{self.N}x{self.M}x{self.frames} {self.title}.pdf' ,dpi = 400)
        # plt.savefig('Retina2.pdf')
        #plt.show()



colors = [
            (1.0, 1.0, 1.0),  # white
            (0.5, 0.7, 0.8),  # pale blue
            (0.1, 0.2, 0.6),  # navy blue
        ]

positions = [0.0, 0.5, 1]  # white at 0, pale blue at 0.2, navy at 0.5 (clipped early)

color_map = LinearSegmentedColormap.from_list("AbyssBlue", list(zip(positions, colors)))


def data_type(type):
    if type == '1 gaussian':
        dff1, params = create_gaussians(N=32, M=64, frames=35, num_gaus=1, x0=18, y0=32, sd0=6, t0_0=17, sdT0=6)
        title = fr'1 Gaussian $\sigma_x$={params[6]} , $\sigma_T$={params[8]}'

        return dff1, title

    if type == '2 gaussian':
        dff1, params = create_gaussians(N=32, M=64, frames=35, num_gaus=2, x0=14, y0=24, sd0=6, t0_0=24,sdT0=5 \
                                                                        , x1=14 , y1=38, sd1=6, t0_1=12, sdT1=5)

        title = r'2 Gaussians $\Delta$X=$\Delta$T=2.5$\sigma$, $\sigma$=6'

        return dff1, title

    if type == 'plane':
        dff1 = create_patterns(N=64, M=32, frames=35, pattern='plane', x0=16, y0=55, sd0=8, u=0, v=1, rad_spd=1,
                               rad_width=3)
        # dff1 = create_patterns(N=64, M=64, frames=80, pattern='plane', x0=-5, y0=0, sd0=8, u=0, v=0.5 ,rad_spd=1, rad_width=4)
        # dff1 = create_patterns(N=128, M=128, frames=110, pattern='plane', x0=-20, y0=0, sd0=8, u=0, v=1 ,rad_spd=1, rad_width=4)

        title = 'Plane Wave'

        return dff1, title

    if type == 'radial':
        #dff1 = create_patterns(N=128, M=64, frames=23, pattern='radial', x0=32, y0=80, sd0=2, u=0, v=1, rad_spd=1,rad_width=5)

        #dff1 = create_patterns(N=80, M=80, frames=30, pattern='radial', x0=40, y0=40, sd0=2, u=0, v=1, rad_spd=1,rad_width=3)

       #  dff1 = create_patterns(N=128, M=64, frames=50, pattern='radial', x0=30, y0=138, sd0=2, u=0, v=1,rad_spd=3, rad_width=5)

        dff1 = create_patterns(N=64, M=32, frames=35, pattern='radial', x0=16, y0=32, sd0=2, u=0, v=0.5,rad_spd=0.5, rad_width=2.5)

        title = 'Radial Wave'

        return dff1, title


def cyclic_hsv_cmap(n=256, rotation=0.25):
    """
    rotation: fraction of the hue cycle to rotate
              0.0 → default (0 angle = red)
              0.25 → rotate by 1/4 (π/2) so red corresponds to top)
    """
    hue = np.linspace(0, 1, n + 1)
    hue = (hue + rotation) % 1  # rotate hue

    saturation = 0.7 + 0.3 * np.sin(2 * np.pi * hue)
    value = 0.85 + 0.15 * np.cos(2 * np.pi * hue)

    hsv = np.stack([hue, saturation, value], axis=1)
    rgb = hsv_to_rgb(hsv.reshape(1, -1, 3)).squeeze()
    rgb = rgb[:-1]  # drop duplicate

    return LinearSegmentedColormap.from_list("cyclic_hsv", rgb)


def phase_cmap():
    smooth_cycle = mcolors.LinearSegmentedColormap.from_list(
        "smooth_cycle",
        [
            "#d45568",
            "#9d4edd",
            "#6c6fbf",
            "#3485a8",
            "#2a9d8f",
            "#8ab17d",
            "#e9c46a",
            "#f4a261",
            "#e76f51",
            "#d45568",  # bridge
        ],
        N=1024
    )

    return smooth_cycle



def classify_jacobian_patterns(flow,plane_threshold=0.8,min_radius=4,min_duration=2,Nv=8,alpha=3.6,beta=0.3):

    """
    Pattern indices:
    0: Plane Wave
    1: Source
    2: Sink
    3: Saddle
    4: Standing Wave
    """
    N, M, T, _ = flow.shape
    delta_1 = alpha * 2 * np.pi / Nv
    delta_2 = beta * 2 * np.pi

    pattern_presence = np.zeros((5, T), dtype=int)
    raw_detections = [[] for _ in range(T)]


    def bilinear_coeffs(f00, f10, f01, f11):
        return f00, f10 - f00, f01 - f00, f11 - f10 - f01 + f00

    def solve_bilinear_intersection(u_vals, v_vals):
        a1, b1, c1, d1 = bilinear_coeffs(*u_vals)
        a2, b2, c2, d2 = bilinear_coeffs(*v_vals)

        x, y = 0.5, 0.5
        for _ in range(10):
            F1 = a1 + b1 * x + c1 * y + d1 * x * y
            F2 = a2 + b2 * x + c2 * y + d2 * x * y

            J = np.array([[b1 + d1 * y, c1 + d1 * x],
                          [b2 + d2 * y, c2 + d2 * x]])

            try:
                delta = np.linalg.solve(J, -np.array([F1, F2]))
            except np.linalg.LinAlgError:
                return None

            x += delta[0]
            y += delta[1]

            if np.linalg.norm(delta) < 1e-6:
                break

        if 0 <= x <= 1 and 0 <= y <= 1:
            return x, y
        return None

    def get_angle(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-5 or n2 < 1e-5:
            return 0
        return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))

    def poincare_index(u, v, y0, x0, radius, samples):

        angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
        vectors = []

        for theta in angles:
            yy = int(round(y0 + radius * np.sin(theta)))
            xx = int(round(x0 + radius * np.cos(theta)))

            if 0 <= yy < u.shape[0] and 0 <= xx < u.shape[1]:
                vec = np.array([u[yy, xx], v[yy, xx]])
                if np.linalg.norm(vec) > 1e-6:
                    vectors.append(vec)

        if len(vectors) < samples // 2:
            return 0

        total_angle = 0

        for i in range(len(vectors)):
            v1 = vectors[i]
            v2 = vectors[(i + 1) % len(vectors)]

            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]

            total_angle += np.arctan2(cross, dot)

        return np.round(total_angle / (2 * np.pi), 3)

    def jacobian(u, v, y, x):
        du_dx = (u[y, x + 1] - u[y, x - 1]) / 2
        du_dy = (u[y + 1, x] - u[y - 1, x]) / 2
        dv_dx = (v[y, x + 1] - v[y, x - 1]) / 2
        dv_dy = (v[y + 1, x] - v[y - 1, x]) / 2
        return np.array([[du_dx, du_dy],
                         [dv_dx, dv_dy]])

    avg_mags = []

    for t in range(T):
        u_m = flow[:, :, t, 0][brain_mask]
        v_m = flow[:, :, t, 1][brain_mask]
        mags = np.sqrt(u_m ** 2 + v_m ** 2)
        if len(mags) > 0:
            avg_mags.append(np.mean(mags))

    avg_mags = np.array(avg_mags)
    standing_thresh = np.mean(avg_mags) - 2 * np.std(avg_mags)

    for t in range(T):

        u = flow[:, :, t, 0]
        v = flow[:, :, t, 1]

        u_m = u[brain_mask]
        v_m = v[brain_mask]

        mags = np.sqrt(u_m ** 2 + v_m ** 2)
        v_avg = np.mean(mags) if len(mags) > 0 else 0

        # --------------------------------------------------------
        # Standing wave (mutually exclusive with plane)
        # --------------------------------------------------------
        if v_avg < standing_thresh:
            pattern_presence[4, t] = 1
            continue

        else:
            # ----------------------------------------------------
            # Plane wave (homogeneity R)
            # ----------------------------------------------------
            if v_avg > 1e-5:
                sum_vec = np.array([np.sum(u_m), np.sum(v_m)])
                R = np.linalg.norm(sum_vec) / (mags.size * v_avg)
                print(R)
                # print(v_avg, np.round(R,4))

                if R >= plane_threshold:
                    pattern_presence[0, t] = 1

        # --------------------------------------------------------
        # Singularities (sources/sinks/saddles)
        # --------------------------------------------------------

        for i in range(1, N - 2):  # avoid edges for 3x3
            for j in range(1, M - 2):

                # Skip if brain mask is zero in the patch
                if not np.all(brain_mask[i - 1:i + 2, j - 1:j + 2]):
                    continue


                u_cell = [u[i, j], u[i, j + 1], u[i + 1, j], u[i + 1, j + 1]]
                v_cell = [v[i, j], v[i, j + 1], v[i + 1, j], v[i + 1, j + 1]]

                if (np.min(u_cell) > 0 or np.max(u_cell) < 0):
                    continue

                if (np.min(v_cell) > 0 or np.max(v_cell) < 0):
                    continue


                sol = solve_bilinear_intersection(u_cell, v_cell)
                if sol is None:
                    continue

                cx = j + sol[0]
                cy = i + sol[1]

                y0 = int(round(cy))
                x0 = int(round(cx))

                best_w = 0
                best_y, best_x = y0, x0

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:

                        yy = y0 + dy
                        xx = x0 + dx

                        if 0 <= yy < N and 0 <= xx < M:

                            w = poincare_index(u, v, yy, xx, radius=3, samples=16)

                            if abs(w) > abs(best_w):
                                best_w = w
                                best_y = yy
                                best_x = xx


                # Compute Jacobian using central differences
                if not (1 <= i < N - 2 and 1 <= j < M - 2):
                    continue

                J_cells = [
                    jacobian(u, v, i, j),
                    jacobian(u, v, i + 1, j),
                    jacobian(u, v, i, j + 1),
                    jacobian(u, v, i + 1, j + 1)
                ]

                J = np.mean(J_cells, axis=0)

                detJ = np.linalg.det(J)
                traceJ = np.trace(J)

                ptype = None

                if abs(best_w) < 0.8:
                    continue

                if detJ < 0:
                    ptype = "Saddle"

                elif detJ > 0:
                    if traceJ > 0:
                        ptype = "Source"
                    elif traceJ < 0:
                        ptype = "Sink"
                else:
                    continue

                    # ------------------------------------------------
                    # Angular criteria: only for Source/Sink
                    # ------------------------------------------------



                if ptype == "Saddle":
                    raw_detections[t].append({
                        'type': ptype,
                        'pos': np.array([cy, cx])
                    })

                elif ptype in ["Source", "Sink"]:

                    neighbors = []
                    angles = np.linspace(0, 2 * np.pi, Nv, endpoint=False)

                    for theta in angles:
                        yy = int(round(cy + min_radius * np.sin(theta)))
                        xx = int(round(cx + min_radius * np.cos(theta)))

                        if 0 <= yy < N and 0 <= xx < M and brain_mask[yy, xx]:
                            neighbors.append(np.array([u[yy, xx], v[yy, xx]]))

                    if len(neighbors) < Nv:
                        continue

                    valid = True

                    # Criterion 1: nearby vectors should change smoothly
                    for k in range(Nv):
                        if get_angle(neighbors[k], neighbors[(k + 1) % Nv]) > delta_1:
                            valid = False
                            break

                    if not valid:
                        continue

                    # Criterion 2: opposite / diagonal vectors should be approximately anti-parallel
                    for k in range(Nv // 2):
                        opp_angle = get_angle(neighbors[k], neighbors[k + Nv // 2])
                        if abs(opp_angle - np.pi) > delta_2:
                            valid = False
                            break

                    if not valid:
                        continue

                    raw_detections[t].append({
                        'type': ptype,
                        'pos': np.array([cy, cx])
                    })


        from sklearn.cluster import DBSCAN

        positions = np.array([d['pos'] for d in raw_detections[t]])
        types = [d['type'] for d in raw_detections[t]]

        if len(positions) > 0:
            clustering = DBSCAN(eps=1.5, min_samples=1).fit(positions)
            merged = []
            for label in np.unique(clustering.labels_):
                cluster_idx = np.where(clustering.labels_ == label)[0]
                avg_pos = positions[cluster_idx].mean(axis=0)
                t_majority = max([types[k] for k in cluster_idx], key=[types[k] for k in cluster_idx].count)
                merged.append({'pos': avg_pos, 'type': t_majority})
            raw_detections[t] = merged

    type_map = {"Source": 1, "Sink": 2, "Saddle": 3}
    active_tracks = []

    for t in range(T):

        current = raw_detections[t]
        matched = set()

        for track in active_tracks:
            if track['active'] and track['last_frame'] == t - 1:

                found_match = False

                for idx, det in enumerate(current):
                    if idx in matched:
                        continue

                    dist = np.linalg.norm(track['last_pos'] - det['pos'])

                    if det['type'] == track['type'] and dist <= min_radius:
                        track['last_pos'] = det['pos']
                        track['last_frame'] = t
                        track['frames'].append(t)
                        matched.add(idx)
                        found_match = True
                        break

                if not found_match:
                    track['active'] = False

        for idx, det in enumerate(current):
            if idx not in matched:
                active_tracks.append({
                    'type': det['type'],
                    'last_pos': det['pos'],
                    'last_frame': t,
                    'frames': [t],
                    'active': True
                })

    filtered_raw_detections = [[] for _ in range(T)]

    for track in active_tracks:
        # Double check that the type is valid and in our map
        if len(track['frames']) >= min_duration and track['type'] in type_map:
            for f in track['frames']:
                pattern_presence[type_map[track['type']], f] += 1
                filtered_raw_detections[f].append({
                    'type': track['type'],
                    'pos': track['last_pos']})

    # print(filtered_raw_detections)
    np.set_printoptions(threshold=np.inf)
    print(pattern_presence)

    return pattern_presence, filtered_raw_detections


brain_mask = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/brain_mask_64.npy')
brain_mask = brain_mask[:, :32]
brain_mask = np.flipud(brain_mask)

outer_line_rgb = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/outer_line_rgb_64.npy')
outer_line_rgb = outer_line_rgb[:, :32]


data_types = ['radial', 'plane', '1 gaussian', '2 gaussian']
snap_frames = [10, 14, 20, 27]  # frames to snapshot
color_map = color_map  # or your preferred colormap



figsize_cm = (18, 15)
figsize_in = tuple(x / 2.54 for x in figsize_cm)
fig, axes = plt.subplots(4, 10, figsize=figsize_in,gridspec_kw={'width_ratios': [1,1,1,1,1.8,1,1,1,1,1.8]})  # 4 rows × 9 columns

for row_idx, dtype in enumerate(data_types):
    #Get data and preprocess
    dff1, title = data_type(type=dtype)
    data = PreDataProcessing(dff1)
    data = FlowAnalyze(data)
    analytic_signal = hilbert(data.dff, axis=-1)#
    phase = np.angle(analytic_signal)  # shape (N, M, T)
    dff1 = data.dff


    data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase=False)
    frame_labels, raw_detections = classify_jacobian_patterns(data.velocities)

    display = Display(data)
    display.title = f'{dtype}'

    for col_idx, frame in enumerate(snap_frames):
        ax = axes[row_idx, col_idx]
        data.dff[:, :, frame][brain_mask == 0] = 0
        ax.imshow(data.dff[:, :, frame], cmap=color_map, vmin=0, vmax=1)

        overlay = outer_line_rgb[::-1].astype(np.float32)

        flow_frame = data.velocities[:, :, frame, :]
        flow_frame[brain_mask == 0] = 0
        plot_quiver(ax, flow_frame, spacing=4, scale=0.04  , color='black', width=0.006)

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Scatter detected patterns
        pattern_colors = {"Source": "tomato", "Sink": "royalblue", "Saddle": "mediumseagreen"}
        xs, ys, colors = [], [], []
        for det in raw_detections[frame]:
            y, x = det['pos']
            ptype = det['type']
            if ptype in pattern_colors:
                xs.append(x)
                ys.append(y)
                colors.append(pattern_colors[ptype])
                ax.text(x + 2, y + 2, ptype, color=pattern_colors[ptype],
                        fontsize=7, weight='bold', ha='left', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        if len(xs) > 0:
            ax.scatter(xs, ys, s=25, edgecolor='white', linewidth=0.5, c=colors)
        ax.set_xticks([]);
        ax.set_yticks([])

        ax.set_ylim(data.dff.shape[0],0)

    # Last column: raster plot
    ax = axes[row_idx, 4]
    pattern_names =["", "", "", "", ""] #["Plane wave", "Source", "Sink", "Saddle", "Standing Wave"]
    T = frame_labels.shape[1]
    time = np.arange(T) / 13
    im = ax.imshow(frame_labels, aspect='auto', origin='lower', extent=[0, time[-1], -0.5, 4.5], cmap=color_map,interpolation='nearest')
    ax.set_yticks(range(5))
    ax.set_yticklabels(pattern_names)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{dtype} raster")
    ax.set_xlim([0,2.8])


    dff1, title = data_type(type=dtype)
    data = PreDataProcessing(dff1)
    data = FlowAnalyze(data)

    data.dff = data.dff - data.dff.mean()

    analytic_signal = hilbert(data.dff, axis=-1)#
    phase = np.angle(analytic_signal)  # shape (N, M, T)

    dff1 = data.dff
    data.dff = phase


    data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase=True)

    variance = np.var(dff1, axis=2)

    mask2d = brain_mask.squeeze().astype(bool)

    phase_analysis_mask = (mask2d & (variance > beta * 0.25))

    # mask velocities before classification
    data.velocities[~phase_analysis_mask, :, :] = 0
    frame_labels, raw_detections = classify_jacobian_patterns(data.velocities)


    display = Display(data)
    display.title = f'{dtype} phase'

    #Phase
    for col_idx, frame in enumerate(snap_frames, start=5):
        ax = axes[row_idx, col_idx]
        frame_data = data.dff[:, :, frame]
        def phase_cmap():
            smooth_cycle = mcolors.LinearSegmentedColormap.from_list(
                "smooth_cycle",
                [
                    "#d45568",
                    "#9d4edd",
                    "#6c6fbf",
                    "#3485a8",
                    "#2a9d8f",
                    "#8ab17d",
                    "#e9c46a",
                    "#f4a261",
                    "#e76f51",
                    "#d45568",  # bridge
                ],
                N=1024
            )

            return smooth_cycle


        print('sdasdasda\nsd ',data.mask)
        mask2d = brain_mask.squeeze().astype(bool)

        masked_phase = np.ma.masked_where(~phase_analysis_mask,phase[:, :, frame])

        ax.imshow(masked_phase,cmap=phase_cmap(),vmin=-np.pi,vmax=np.pi)





        overlay = outer_line_rgb[::-1].astype(np.float32)
        #ax.imshow(overlay, alpha=(overlay > 0).astype(np.float32),cmap='Greys')

        # Quiver
        flow_frame = data.velocities[:, :, frame, :]

        flow_frame[brain_mask == 0] = 0
        plot_quiver(ax, flow_frame, spacing=4, scale=0.2, color='black', width=0.006)

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Scatter detected patterns
        pattern_colors = {"Source": "tomato", "Sink": "royalblue", "Saddle": "mediumseagreen"}
        xs, ys, colors = [], [], []
        for det in raw_detections[frame]:
            y, x = det['pos']
            ptype = det['type']
            if ptype in pattern_colors:
                xs.append(x)
                ys.append(y)
                colors.append(pattern_colors[ptype])
                ax.text(x + 2, y + 2, ptype, color=pattern_colors[ptype],
                        fontsize=7, weight='bold', ha='left', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        if len(xs) > 0:
            ax.scatter(xs, ys, s=25, edgecolor='white', linewidth=0.5, c=colors)
        ax.set_xticks([]);
        ax.set_yticks([])

        ax.set_ylim(data.dff.shape[0],0)

    # Last column: raster plot
    ax = axes[row_idx, 9]
    pattern_names =["", "", "", "", ""] #["Plane wave", "Source", "Sink", "Saddle", "Standing Wave"]
    T = frame_labels.shape[1]
    time = np.arange(T) / 13
    im = ax.imshow(frame_labels, aspect='auto', origin='lower', extent=[0, time[-1], -0.5, 4.5], cmap=color_map,interpolation='nearest')
    ax.set_yticks(range(5))
    ax.set_yticklabels(pattern_names)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{dtype} raster")
    ax.set_xlim([0,2.8])

plt.tight_layout()
plt.subplots_adjust(wspace=0.6, hspace=0.4)
plt.savefig(f"Jabocian_poincare_detection_alpha={alpha},iter={iterations}_.svg", bbox_inches='tight', transparent=False, dpi=300)

plt.show()


