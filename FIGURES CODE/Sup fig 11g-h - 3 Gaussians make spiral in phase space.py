import numpy as np
import cv2
import scipy
import h5py
from PIL import Image
from scipy.ndimage import binary_erosion
import math
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from scipy.ndimage import minimum_filter
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import map_coordinates
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter, maximum_filter, label
from skimage.morphology import skeletonize
from scipy.ndimage import uniform_filter
from scipy.io import savemat
from scipy.ndimage import convolve
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, hsv_to_rgb
from scipy import ndimage

from Algos.Display import plot_quiver
from scipy.stats import pearsonr

from Algos.Create_Patterns import create_patterns, create_gaussians, create_gaussians_moving
from Algos.Data_Processing import Filter, resize, decrease_frame_rate, normalize_data
from Algos.Horn_Schunck import horn_schunck , horn_schunck_phase
from Algos.Spatial_cohernce import nodes_connections
import scipy.io
from scipy.io import loadmat
import scipy.stats as stats

#### Algorithm parameters###
alpha = 0.3   ## Optic Flow Horn-Schunck alpha parameter
iterations = 150  ## Number of iterations
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

            self.phase_space[:, :, i, 0] = flow[:, :, 0] * self.dff[:, :, i]
            self.phase_space[:, :, i, 1] = flow[:, :, 1] * self.dff[:, :, i]

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

    def calculate_waveness(self, type):

        self.abs_phase_space = np.abs(self.phase_space)

        self.sum_phase_space[:, :, :, 0] = np.cumsum(self.phase_space[:, :, :, 0], axis=2)
        self.sum_phase_space[:, :, :, 1] = np.cumsum(self.phase_space[:, :, :, 1], axis=2)

        self.abs_sum_phase_space[:, :, :, 0] = np.cumsum(self.abs_phase_space[:, :, :, 0], axis=2)
        self.abs_sum_phase_space[:, :, :, 1] = np.cumsum(self.abs_phase_space[:, :, :, 1], axis=2)

        variance = np.var(self.dff, axis=2)

        self.spatial_coh = []
        self.temporal_coh = []
        self.spatial_temporal_coh = []

        up = np.linalg.norm(self.sum_phase_space[..., -1, :], axis=-1)
        down = np.linalg.norm(self.abs_sum_phase_space[..., -1, :], axis=-1)
        ratios = np.divide(up, down, where=down != 0, out=np.zeros_like(up))

        # neighborhood_size = ((2 * self.n + 1) ** 2 - 4 * (2 * self.n))   ## First order
        neighborhood_size = (2 * self.n + 1) ** 2 - (4 * 2 * self.n) - (4 * (2 * n - 2))  ## Second order

        brain_mask = np.load('brain_mask.npy')
        if self.dff.shape[1] == 64:
            brain_mask = brain_mask[:, :64]

        valid_mask = variance > (beta * 0.25)

        activated_pixels = np.count_nonzero(valid_mask)
        brain_activity_area = 12546  # for 128x128


        filtered_map = np.where(valid_mask, variance, 0)

        def find_max_min(data, sigma, prominence, lim_up, lim_down, height=None, distance=5):
            """
            Applies Gaussian smoothing to the input data and finds strictly positive maxima (peaks)
            and strictly negative minima (troughs).

            Args:
                data: 1D array of gradient values.
                sigma: Standard deviation for the Gaussian filter.
                prominence: Minimum prominence of peaks.
                height: Minimum height of peaks.
                distance: Minimum distance between peaks.

            Returns:
                1 if exactly one strictly positive peak appears before exactly one strictly negative trough, otherwise 0.
            """
            # Apply Gaussian smoothing
            smoothed_data = gaussian_filter1d(data, sigma=sigma)

            # Example inputs:
            # data_cube: (N, M, T)
            # mask: (N, M), boolean
            # Make sure mask is True where activity exists

            # Temporal gradient

            # print(f"Adaptive limit = {lim:.5f}")

            # Find peaks (maxima)
            peaks, _ = find_peaks(smoothed_data, prominence=prominence, height=height, distance=distance)
            peaks = [t for t in peaks if smoothed_data[t] > lim_up]

            peaks_raw, _ = find_peaks(data, prominence=prominence, height=height, distance=distance)
            peaks_raw = [t for t in peaks_raw if data[t] > lim_up]

            # Find troughs (minima)
            troughs, _ = find_peaks(-smoothed_data, prominence=prominence, height=height, distance=distance)
            troughs = [t for t in troughs if smoothed_data[t] < lim_down]

            troughs_raw, _ = find_peaks(-data, prominence=prominence, height=height, distance=distance)
            troughs_raw = [t for t in troughs_raw if data[t] < lim_down]

            # Check if there is exactly one peak and one trough, and peak appears before trough
            if len(peaks) == 1 and len(troughs) == 1 and troughs[0] < peaks[0]:
                # print("normal")
                # print(peaks , smoothed_data[peaks[0]])
                # print(troughs, smoothed_data[troughs[0]])
                return 1
                # elif len(peaks_raw) == 1 and len(troughs_raw) == 1 and troughs_raw[0] < peaks_raw[0]:
                # print("raw")
                # print(peaks_raw)
                # print(troughs_raw)
                return 1

            return 0

        def analyze_gradients_in_rectangle(frame, gradient_map, x, y, direction, rect_size, t, velocities, lim_up,
                                           lim_down, sigma=1, prominence=0, boundary_condition=False):
            """
            Analyze if there is a maximum and minimum gradient in a rotated rectangle centered at (x, y),
            aligned along `direction`, and applies Gaussian smoothing before checking peaks and troughs.

            Args:
                gradient_map: 2D NumPy array of gradient magnitudes.
                x, y: Center pixel coordinates.
                direction: Velocity vector (vx, vy).
                rect_size: (height, width) of the rectangle.
                sigma: Standard deviation for the Gaussian smoothing.
                prominence: Prominence of peaks for peak detection.

            Returns:
                1 if both maximum and minimum gradients are found, otherwise 0.
            """
            H, W = gradient_map.shape
            rect_h, rect_w = rect_size

            # Normalize direction vector
            vx, vy = direction
            norm = np.hypot(vx, vy)
            if norm < 0.001:
                return 0  # No movement -> No rectangle

            unit_vx, unit_vy = vx / norm, vy / norm

            # Compute rotated rectangle corners
            dx_w = (rect_w / 2) * unit_vy  # Perpendicular to velocity
            dy_w = -(rect_w / 2) * unit_vx

            dx_h = (rect_h / 2) * unit_vx  # Along velocity direction
            dy_h = (rect_h / 2) * unit_vy

            corners = np.array([
                [x - dx_w - dx_h, y - dy_w - dy_h],  # Top-left
                [x + dx_w - dx_h, y + dy_w - dy_h],  # Top-right
                [x + dx_w + dx_h, y + dy_w + dy_h],  # Bottom-right
                [x - dx_w + dx_h, y - dy_w + dy_h]  # Bottom-left
            ], dtype=np.float32)

            # Ensure valid integer coordinates
            corners = np.clip(corners, [0, 0], [W - 1, H - 1]).astype(np.int32)

            # Create a binary mask
            mask = np.zeros((H, W), dtype=np.uint8)  # Ensure it's correctly shaped
            cv2.fillPoly(mask, [corners.reshape((-1, 1, 2))], 1)

            # Apply the mask to the gradient map
            masked_gradients = gradient_map * mask

            # Count how many masked pixels are zero
            num_zeros = np.sum((mask == 1) & (masked_gradients == 0))

            if num_zeros > 0 and boundary_condition == True:
                max_extension = int(rect_h / 2)
                step_size = 1

                directions = {
                    'backward': (-unit_vx, -unit_vy),
                    'forward': (unit_vx, unit_vy),
                    'left': (unit_vy, -unit_vx),  # Perpendicular
                    'right': (-unit_vy, unit_vx),  # Perpendicular other side
                }

                for step in range(1, max_extension + 1):
                    best_shift = None
                    best_num_zeros = num_zeros
                    best_corners = None
                    best_mask = None
                    best_masked_gradients = None

                    for dir_name, (dx, dy) in directions.items():
                        # Skip perpendicular directions if step > 4
                        if dir_name in ["left", "right"] and step > 4:
                            continue

                        shift_x = dx * step * step_size
                        shift_y = dy * step * step_size

                        x_shifted = x + shift_x
                        y_shifted = y + shift_y

                        dx_h = (rect_h / 2) * unit_vx
                        dy_h = (rect_h / 2) * unit_vy

                        test_corners = np.array([
                            [x_shifted - dx_w - dx_h, y_shifted - dy_w - dy_h],
                            [x_shifted + dx_w - dx_h, y_shifted + dy_w - dy_h],
                            [x_shifted + dx_w + dx_h, y_shifted + dy_w + dy_h],
                            [x_shifted - dx_w + dx_h, y_shifted - dy_w + dy_h],
                        ], dtype=np.float32)

                        test_corners = np.clip(test_corners, [0, 0], [W - 1, H - 1]).astype(np.int32)

                        test_mask = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(test_mask, [test_corners.reshape((-1, 1, 2))], 1)

                        test_masked_grad = gradient_map * test_mask
                        new_zeros = np.sum((test_mask == 1) & (test_masked_grad == 0))

                        if new_zeros < best_num_zeros:
                            best_num_zeros = new_zeros
                            best_shift = (shift_x, shift_y)
                            best_corners = test_corners
                            best_mask = test_mask
                            best_masked_gradients = test_masked_grad

                    if best_num_zeros < num_zeros:
                        corners = best_corners
                        mask = best_mask
                        masked_gradients = best_masked_gradients
                        num_zeros = best_num_zeros

                    if num_zeros == 0:
                        break

            def bresenham_line(x0, y0, x1, y1):
                """Bresenham’s Line Algorithm to get pixel coordinates between two points."""
                points = []
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                sx = 1 if x0 < x1 else -1
                sy = 1 if y0 < y1 else -1
                err = dx - dy

                while True:
                    points.append((x0, y0))
                    if x0 == x1 and y0 == y1:
                        break
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x0 += sx
                    if e2 < dx:
                        err += dx
                        y0 += sy

                return points

            # Extract start and end points for the line along direction
            start_x, start_y = corners[0]  # Top-left corner as reference
            end_x, end_y = corners[2]  # Bottom-right as reference

            line_pixels = bresenham_line(start_x, start_y, end_x,
                                         end_y)  # Use Bresenham’s algorithm to get the pixel coordinates along the detected line

            extracted_values = [gradient_map[y, x] for x, y in line_pixels if
                                0 <= x < W and 0 <= y < H]  # Extract values from gradient_map at those locations

            profile_1d = np.array(extracted_values)
            profile_1d = profile_1d[profile_1d != 0]  # Exclude zero values

            masked_gradients_nan = np.where(masked_gradients == 0, np.nan, masked_gradients)

            # Call the find_max_min function on the smoothed gradients
            score = find_max_min(profile_1d, sigma=sigma, prominence=prominence, lim_up=lim_up, lim_down=lim_down)
            if score == 10:
                figsize_cm = (22, 12)
                figsize_in = tuple(x / 2.54 for x in figsize_cm)
                fig, axes = plt.subplots(2, 3, figsize=figsize_in)

                im0 = axes[0, 0].imshow(frame, cmap='Blues', vmin=0, vmax=1)
                cbar0 = fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
                cbar0.ax.tick_params(labelsize=10)
                cbar0.set_ticks([0, 1])
                cbar0.set_ticklabels(['0', '1'])
                axes[0, 1].imshow(frame, cmap='Blues', vmin=0, vmax=1)
                plot_quiver(axes[0, 1], self.phase_space[:, :, t, :], spacing=4, scale=0.05, color='black')
                axes[0, 1].set_ylim(frame.shape[0], 0)

                dff_gradient = np.gradient(self.dff, axis=2)  # Time derivative of dff
                im2 = axes[0, 2].imshow(dff_gradient[:, :, 16], cmap="RdBu", vmin=-0.15, vmax=0.15)
                cbar2 = fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
                cbar2.ax.tick_params(labelsize=10)
                cbar2.set_ticks([-0.15, 0, 0.15])
                cbar2.set_ticklabels(['-0.15', '0', '0.15'])

                axes[1, 0].imshow(frame, cmap='Blues', vmin=0, vmax=1)
                plot_quiver(axes[1, 0], self.phase_space[:, :, t, :], spacing=4, scale=0.05, color='black')
                axes[1, 0].scatter(x, y, color='darkviolet', s=30)
                axes[1, 0].imshow(masked_gradients_nan, cmap='RdBu', vmin=-0.15, vmax=0.15)
                axes[1, 0].set_aspect('auto')
                axes[1, 0].set_ylim(frame.shape[0], 0)

                smoothed_data = gaussian_filter1d(profile_1d, sigma=sigma)

                axes[1, 1].plot(profile_1d, marker='o', linestyle='-', color='royalblue', label='1D Signal ',
                                markersize=4)
                axes[1, 1].set_ylim([-0.25, 0.25])

                axes[1, 1].set_ylabel("Time Derivative")
                axes[1, 1].set_xlabel("Index")

                # Plot the 1D flattened gradients
                axes[1, 2].plot(profile_1d, marker='o', linestyle='-', color='royalblue', markersize=4, alpha=0.8,
                                linewidth=1.5)
                axes[1, 2].plot(smoothed_data, marker=' ', linestyle='-', color='brown', label='Smoothed 1D Signal ',
                                markersize=4, linewidth=1.5)

                axes[1, 2].axhline(y=lim_down, color='black', linestyle='--', linewidth=1)  # First line at y=-0.0429
                axes[1, 2].axhline(y=lim_up, color='black', linestyle='--', linewidth=1)  # Second line at y=0.0439
                print(lim)

                # axes[1,2].set_title("Flattened 1D Time Derivative ")
                axes[1, 2].set_xlabel("Index")
                axes[1, 2].set_ylim([-0.15, 0.15])
                # axes[1,1].legend(loc='upper left', fontsize=8)

                handles, labels = axes[1, 1].get_legend_handles_labels()
                fig.legend(
                    handles, labels,
                    loc='lower center',  # or 'upper center', depending on your layout
                    bbox_to_anchor=(0.5, 0.47),  # center horizontally, slightly below the figure
                    frameon=False,
                    fontsize=8,
                    ncol=3,  # <-- this makes it a single row
                    handlelength=2.3
                )

                cbar0 = fig.colorbar(im0, ax=axes[0, 1], fraction=0.046, pad=0.04)
                cbar0.set_ticks([-0.15, 0, 0.15])
                cbar0.set_ticklabels(['-0.15', '0', '0.15'])
                cbar0 = fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)
                cbar0.set_ticks([-0.15, 0, 0.15])
                cbar0.set_ticklabels(['-0.15', '0', '0.15'])
                cbar0 = fig.colorbar(im0, ax=axes[1, 1], fraction=0.046, pad=0.04)
                cbar0.set_ticks([-0.15, 0, 0.15])
                cbar0.set_ticklabels(['-0.15', '0', '0.15'])
                cbar0 = fig.colorbar(im0, ax=axes[1, 2], fraction=0.046, pad=0.04)
                cbar0.set_ticks([-0.15, 0, 0.15])
                cbar0.set_ticklabels(['-0.15', '0', '0.15'])

                handles, labels = axes[1, 2].get_legend_handles_labels()
                first_legend = plt.legend([handles[0]], [labels[0]],
                                          loc='upper center',
                                          bbox_to_anchor=(0.5, 1.23),
                                          frameon=False,
                                          fontsize=8)

                # Second legend: the other two entries
                second_legend = plt.legend(handles[1:], labels[1:],
                                           loc='upper center',
                                           bbox_to_anchor=(0.5, 1.14),
                                           frameon=False,
                                           fontsize=8,
                                           ncol=2)

                plt.gca().add_artist(first_legend)  # keep the first legend

                plt.tight_layout()

                plt.savefig("figure.pdf", format="pdf", dpi=600, bbox_inches='tight')
                plt.show()

            return score


        def process_data(frames, velocities, search_map, rect_size=(34, 4)):
            """
            Process a video sequence, summing gradients within directional rectangles.

            Args:
                frames: 3D NumPy array of shape (H, W, T) representing grayscale frames.
                velocities: 4D NumPy array of shape (H, W, T, 2), containing (vx, vy) per pixel.
                search_map: 2D NumPy array (H, W), where positive values indicate pixels to process.
                rect_size: Tuple (height, width) of the rectangle.

            Returns:
                A 3D array (H, W, T) with summed gradients for each pixel.
            """

            H, W, T = frames.shape
            wave_front_map = np.zeros((H, W))

            # Get the indices where search_map > 0
            search_indices = np.argwhere(search_map > 0)
            # search_indices = [(85,85)]

            grad_t = np.gradient(frames, axis=2)  # (N, M, T)

            # Use only masked pixels
            if grad_t.shape[:len(brain_mask.shape)] == brain_mask.shape:
                masked_grad = grad_t[brain_mask, :]  # Apply mask
            else:
                masked_grad = grad_t  # No masking

            # Compute average + std over time and pixels
            avg_grad = masked_grad.mean()
            std_grad = masked_grad.std()
            lim_up = avg_grad + std_grad
            lim_down = avg_grad - std_grad
            # print("avg grad: ",np.round(avg_grad,4)," std grad: ",np.round(std_grad,4), " lim_up: ",np.round(lim_up,4) , " lim_down: ",np.round(lim_down,4))

            for t in range(0, T - 1):  # Ensure valid temporal indexing
                frame = frames[:, :, t]
                gradient_map = frames[:, :, t + 1] - frame

                velocity_map = velocities[:, :, t, :]  # Shape: (H, W, 2)

                for y, x in search_indices:  # Iterate only through relevant pixels
                    if wave_front_map[y, x] == 1:  # Skip if any time point at (y,x) is already 1
                        continue

                    direction = velocity_map[y, x, :]  # Safe access
                    wave_front_map[y, x] = analyze_gradients_in_rectangle(frame, gradient_map, x, y, direction,
                                                                          rect_size, t, velocities, lim_up, lim_down)

            return wave_front_map

        wave_front_map = process_data(self.dff, self.velocities, filtered_map)

        for h in range(0, self.N):
            for j in range(0, self.M):

                ## Top left corner
                if (h < self.n) and (j < self.n):
                    vector_field = self.sum_phase_space[0:2 * (self.n) + 1, 0:2 * (self.n) + 1, -1, :]

                ## Top side
                elif (h < self.n) and (self.n <= j < self.M - self.n):
                    vector_field = self.sum_phase_space[0:2 * (self.n) + 1, j - self.n:j + self.n + 1, -1, :]

                ## Top right corner
                elif (h < self.n) and (self.M - self.n <= j):
                    vector_field = self.sum_phase_space[0:2 * (self.n) + 1, self.M - (2 * (self.n) + 1):self.M, -1, :]

                ## Bottom Left corner
                elif (h >= self.N - self.n) and (j <= self.n):
                    vector_field = self.sum_phase_space[self.N - (2 * (self.n) + 1):self.N, 0:2 * (self.n) + 1, -1, :]

                ## Bottom side
                elif (h >= self.N - self.n) and (self.n <= j < self.M - self.n):
                    vector_field = self.sum_phase_space[self.N - (2 * (self.n) + 1):self.N, j - self.n:j + self.n + 1,
                                   -1, :]

                ## Bottom right corner
                elif (h >= self.N - self.n) and (j >= self.M - self.n):
                    vector_field = self.sum_phase_space[self.N - (2 * (self.n) + 1):self.N,
                                   self.M - (2 * (self.n) + 1):self.M, -1, :]

                ## Left side
                elif (j < self.n) and (self.n <= h <= self.N - self.n):
                    vector_field = self.sum_phase_space[h - self.n:h + self.n + 1, 0:2 * (self.n) + 1, -1, :]

                ## Right side
                elif (j >= self.M - self.n) and (self.n <= h < self.N - self.n):
                    vector_field = self.sum_phase_space[h - self.n:h + self.n + 1, self.M - (2 * (self.n) + 1):self.M,
                                   -1, :]

                ## Center
                else:
                    vector_field = self.sum_phase_space[h - self.n:h + self.n + 1, j - self.n:j + self.n + 1, -1, :]

                h_start = max(h - (n - 2), 0)
                h_end = min(h + (n - 2) + 1, variance.shape[0])
                j_start = max(j - (n - 2), 0)
                j_end = min(j + (n - 2) + 1, variance.shape[1])

                def global_directional_coherence(field):
                    # Extract U and V components
                    U = field[..., 0]
                    V = field[..., 1]

                    # Compute angle of each vector
                    theta = np.arctan2(V, U).flatten()

                    # Ignore zero vectors (undefined direction)
                    magnitude = np.hypot(U, V).flatten()
                    valid = magnitude > 1e-6
                    theta_valid = theta[valid]

                    # Compute mean resultant length
                    R = np.abs(np.mean(np.exp(1j * theta_valid)))

                    return R

                # spatial_coherence=global_directional_coherence(vector_field)
                # print(h,j,spatial_coherence)

                if variance[h, j] > beta * 0.25 and (type != 'cortex' or brain_mask[h, j] != 0):  # and spatial_coherence>0.5:
                    sum_vx = np.sum(vector_field[..., 0])
                    sum_vy = np.sum(vector_field[..., 1])

                    avg_vx = sum_vx / ((2 * self.n) ** 2)
                    avg_vy = sum_vy / ((2 * self.n) ** 2)

                    angles = np.arctan2(avg_vx, avg_vy)
                    color = mcolors.hsv_to_rgb([(angles + np.pi) / (2 * np.pi), 1, 1])  # Saturation and Value are 1

                    # self.waveness[h, j, 0] = color[0]
                    self.waveness[h, j, 0] = angles
                    self.waveness[h, j, 1] = color[1]
                    self.waveness[h, j, 2] = color[2]

                    self.waveness[h, j, 3] = ratios[h, j] * wave_front_map[h, j]  # * spatial_coherence

                    self.mask[h, j] = 1


                else:
                    self.waveness[h, j, 3] = 0

        self.Waveness = True


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

    def plot_frames_hilbert(self, space, scale):
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

        index = 0
        # frame_indices = [index, index + 5, index + 10, index + 21, index + 30, index + 45,
        #                 index + 51, index + 55, index + 61, index + 63, index + 67]

        # frame_indices = [9,18, 26,37,41]
        # frame_indices = [3,5,7,9,11]
        frame_indices = [1, 4, 6, 8, 10, 12, 14]

        outer_line_rgb = np.load('outer_line_rgb.npy')
        outer_line_rgba = np.ones((*outer_line_rgb.shape[:2], 4))
        line_mask = np.all(outer_line_rgb < 1.0, axis=2)
        outer_line_rgba[..., :3][line_mask] = 0.392
        outer_line_rgba[..., 3] = 0.0
        outer_line_rgba[..., 3][line_mask] = 1.0

        brain_mask = np.load('brain_mask.npy')

        mask = (brain_mask == 1)

        # Zero alpha outside brain
        outer_line_rgba[..., 3][~mask] = 0.0

        # (optional) also zero color outside brain so nothing shows
        outer_line_rgba[..., :3][~mask] = 0.0

        figsize_cm = (12, 9)
        figsize_in = tuple(x / 2.54 for x in figsize_cm)
        fig = plt.figure(figsize=figsize_in, dpi=350)

        cols = len(frame_indices)
        gs = gridspec.GridSpec(
            3, cols + 1,
            width_ratios=[1] * cols + [0.03],
            height_ratios=[1, 1, 1],  # ← 3 rows now
            hspace=0.2, wspace=0)

        # Single row for each: Calcium and Flow
        top_axes = [fig.add_subplot(gs[0, i]) for i in range(cols)]
        hilbert_axes = [fig.add_subplot(gs[1, i]) for i in range(cols)]



        for i, frame_idx in enumerate(frame_indices):
            print(i, frame_idx)
            snapshot = self.dff[:, :, frame_idx]

            ax_top = top_axes[i]
            ax_top.imshow(dff1[:, :, frame_idx], cmap=self.color_map, vmin=0, vmax=1)
            ax_top.imshow(outer_line_rgba[:, :64])
            ax_top.set_aspect('equal')
            ax_top.axis('off')



            ax_hilbert = hilbert_axes[i]
            frame = data.dff[:, :, frame_idx].copy()

            # Set everything outside the brain mask to white
            frame[brain_mask[:, :64] == 0] = 1.0  # white for your colormap
            self.flows[frame_idx][:, :, 0] = np.where(brain_mask[:, :64] == 1, self.flows[frame_idx][:, :, 0], 0)
            self.flows[frame_idx][:, :, 1] = np.where(brain_mask[:, :64] == 1, self.flows[frame_idx][:, :, 1], 0)

            ax_hilbert.imshow(frame, cmap=phase_cmap(), vmin=-np.pi, vmax=np.pi)
            plot_quiver(ax_hilbert, self.flows[frame_idx], spacing=space, scale=scale, color='black', width=0.006)
            overlay = np.ones((*frame.shape, 4))  # R=G=B=A=1
            overlay[..., 3] = 1.0  # fully opaque

            # Apply brain mask: make overlay transparent inside brain
            mask = (brain_mask[:, :frame.shape[1]] == 1)
            overlay[mask, 3] = 0.0  # alpha=0 inside brain

            ax_top.imshow(overlay, cmap=self.color_map, vmin=0, vmax=1)

            # Overlay it
            ax_hilbert.imshow(overlay)
            ax_hilbert.imshow(outer_line_rgba[:, :64])
            ax_hilbert.invert_yaxis()
            ax_hilbert.set_aspect('equal')
            ax_hilbert.axis('off')
            ax_hilbert.set_ylim(frame.shape[0], 0)

        fig.text(0.145, 0.92, "a 3 Gaussians 0–0.4s, 35Hz", ha='left', va='top', fontsize=5, font='Arial')
        fig.text(0.145, 0.65, "b Hilbert Space & Optic Flow", ha='left', va='top', fontsize=5, font='Arial')

        plt.savefig('Ca_OF_combined_rows.pdf', bbox_inches='tight', dpi=600)
        plt.show()


def data_type(type):

    if type == '3 gaussian':
        # dff1 , params = create_gaussians(N=64, M=64, frames=50, num_gaus=2, x0=20+17.5, y0=32, sd0=7, t0_0=19, sdT0=7 \
        #                                                        , x1=20, y1=32,sd1=7, t0_1=19+17.5 ,sdT1=7)

        dff1, params = create_gaussians(N=64, M=128, frames=28, num_gaus=1, x0=46, y0=85, sd0=12, t0_0=4, sdT0=2.3,
                                         x1=20, y1=90, sd1=7, t0_1=19 + 17.5, sdT1=7)
        dff2, params = create_gaussians(N=64, M=128, frames=28, num_gaus=1, x0=46, y0=55, sd0=12, t0_0=8, sdT0=2.3,
                                        x1=20, y1=90, sd1=7, t0_1=19 + 17.5, sdT1=7)
        dff3, params = create_gaussians(N=64, M=128, frames=28, num_gaus=1, x0=20, y0=70, sd0=12, t0_0=12, sdT0=2.3,
                                        x1=20, y1=90, sd1=7, t0_1=19 + 17.5, sdT1=7)
        #dff4, params = create_gaussians(N=64, M=128, frames=35, num_gaus=1, x0=46, y0=85, sd0=12, t0_0=16, sdT0=2.3,
        #                                x1=20, y1=90, sd1=7, t0_1=19+17.5, sdT1=7)
        # dff5 , params = create_gaussians(N=64, M=128, frames=35, num_gaus=1, x0=46, y0=55, sd0=12, t0_0=20, sdT0=2.3,
        #                                x1=20, y1=90, sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff6 , params = create_gaussians(N=64, M=128, frames=35, num_gaus=1, x0=20, y0=70, sd0=12, t0_0=24, sdT0=2.3,
        #                                 x1=20, y1=90, sd1=7, t0_1=19+17.5 ,sdT1=7)

        # dff1 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=44, y0=82.5, sd0=10, t0_0=8, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff2 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=44, y0=60, sd0=10, t0_0=18, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff3 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=24.52, y0=71.25, sd0=10, t0_0=28, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff4 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=44, y0=82.5, sd0=10, t0_0=38, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff5 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=46, y0=55, sd0=12, t0_0=48, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff6 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=20, y0=70, sd0=12, t0_0=58, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)

        title = r'3 Gaussians '

        dff = dff1 + dff2 + dff3  # + dff4 #+ dff5 + dff6 #+dff7 +dff8

        return dff, title

    if type == '3 gaussian science edtior':
        # dff1 , params = create_gaussians(N=64, M=64, frames=50, num_gaus=2, x0=20+17.5, y0=32, sd0=7, t0_0=19, sdT0=7 \
        #                                                        , x1=20, y1=32,sd1=7, t0_1=19+17.5 ,sdT1=7)

        dff1, params = create_gaussians(N=128, M=128, frames=50, num_gaus=1, x0=75, y0=76, sd0=12, t0_0=9, sdT0=2.3,
                                         x1=20, y1=90, sd1=7, t0_1=19 + 17.5, sdT1=7)
        dff2, params = create_gaussians(N=128, M=128, frames=50, num_gaus=1, x0=45, y0=76, sd0=12, t0_0=13, sdT0=2.3,
                                        x1=20, y1=90, sd1=7, t0_1=19 + 17.5, sdT1=7)
        dff3, params = create_gaussians(N=128, M=128, frames=50, num_gaus=1, x0=60, y0=50, sd0=12, t0_0=17, sdT0=2.3,
                                        x1=20, y1=90, sd1=7, t0_1=19 + 17.5, sdT1=7)
        #dff4, params = create_gaussians(N=64, M=128, frames=35, num_gaus=1, x0=46, y0=85, sd0=12, t0_0=16, sdT0=2.3,
        #                                x1=20, y1=90, sd1=7, t0_1=19+17.5, sdT1=7)
        # dff5 , params = create_gaussians(N=64, M=128, frames=35, num_gaus=1, x0=46, y0=55, sd0=12, t0_0=20, sdT0=2.3,
        #                                x1=20, y1=90, sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff6 , params = create_gaussians(N=64, M=128, frames=35, num_gaus=1, x0=20, y0=70, sd0=12, t0_0=24, sdT0=2.3,
        #                                 x1=20, y1=90, sd1=7, t0_1=19+17.5 ,sdT1=7)

        # dff1 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=44, y0=82.5, sd0=10, t0_0=8, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff2 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=44, y0=60, sd0=10, t0_0=18, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff3 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=24.52, y0=71.25, sd0=10, t0_0=28, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff4 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=44, y0=82.5, sd0=10, t0_0=38, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff5 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=46, y0=55, sd0=12, t0_0=48, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)
        # dff6 , params = create_gaussians(N=64, M=128, frames=65, num_gaus=1, x0=20, y0=70, sd0=12, t0_0=58, sdT0=4 , x1=20, y1=90,sd1=7, t0_1=19+17.5 ,sdT1=7)

        title = r'3 Gaussians '

        dff = dff1 + dff2 + dff3  # + dff4 #+ dff5 + dff6 #+dff7 +dff8

        return dff, title

def bandpass_filter_video(video, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')

    # Apply filter along the last axis (time)
    filtered = filtfilt(b, a, video, axis=-1)

    return filtered



brain_mask = np.load('brain_mask.npy')


brain_mask = brain_mask


dff1, title = data_type('3 gaussian')

data = PreDataProcessing(dff1)

fs = 35.0  # frames per second (adjust to your case)
lowcut = 2
highcut = 8.0

filtered_dff = bandpass_filter_video(data.dff, lowcut, highcut, fs)
analytic_signal = hilbert(filtered_dff, axis=-1)  #
phase = np.angle(analytic_signal)  # shape (N, M, T)
data.dff = phase

data = FlowAnalyze(data)

data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase=True)
#data.calculate_waveness(type='retina')
display = Display(data)
display.title = 'Fig 2 example'
display.plot_frames_hilbert( space = 6 , scale = 1)

