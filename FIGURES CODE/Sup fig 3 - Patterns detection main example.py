import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import matplotlib
from scipy.ndimage import gaussian_filter, label
from sklearn.cluster import DBSCAN
from Algos.Display import plot_quiver
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import hsv_to_rgb
from Algos.Data_Processing import Filter, resize, decrease_frame_rate, normalize_data
from Algos.Horn_Schunck import horn_schunck, horn_schunck_phase

#### Algorithm parameters###
alpha = 0.5  ## Optic Flow Horn-Schunck alpha parameter
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

    def horn_schunck_flow(self, alpha, num_iter, phase):
        all_convergence = []

        for i in range(self.frames - 1):
            image1 = self.dff[:, :, i]
            image2 = self.dff[:, :, i + 1]

            if phase == True:
                flow, convergence = horn_schunck_phase(image1, image2, alpha, num_iter)
            else:
                flow, convergence = horn_schunck(image1, image2, alpha, num_iter)

            #### If phase map

            # phase1 = np.mod(self.dff[:, :, i], 2 * np.pi)
            # phase2 = np.mod(self.dff[:, :, i + 1], 2 * np.pi)
            # image1 = np.exp(1j * phase1)
            # image2 = np.exp(1j * phase2)
            # flow, convergence = horn_schunck(image1, image2, alpha, num_iter)

            self.flows.append(flow)

            self.velocities[:, :, i, 0] = flow[:, :, 0]
            self.velocities[:, :, i, 1] = flow[:, :, 1]

            self.space_and_time[:, :, i, 0] = self.dff[:, :, i]
            self.space_and_time[:, :, i, 1] = flow[:, :, 0]
            self.space_and_time[:, :, i, 2] = flow[:, :, 1]

            self.phase_space[:, :, i, 0] = flow[:, :, 0]  # * self.dff[:, :, i]
            self.phase_space[:, :, i, 1] = flow[:, :, 1]  # * self.dff[:, :, i]

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
                plot_quiver(axes[0, 1], data.phase_space[:, :, t, :], spacing=4, scale=0.05, color='black')
                axes[0, 1].set_ylim(frame.shape[0], 0)

                dff_gradient = np.gradient(self.dff, axis=2)  # Time derivative of dff
                im2 = axes[0, 2].imshow(dff_gradient[:, :, 16], cmap="RdBu", vmin=-0.15, vmax=0.15)
                cbar2 = fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
                cbar2.ax.tick_params(labelsize=10)
                cbar2.set_ticks([-0.15, 0, 0.15])
                cbar2.set_ticklabels(['-0.15', '0', '0.15'])

                axes[1, 0].imshow(frame, cmap='Blues', vmin=0, vmax=1)
                plot_quiver(axes[1, 0], data.phase_space[:, :, t, :], spacing=4, scale=0.05, color='black')
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

        ## y=77,x=41,t=31 for example of radial wave

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

        # wave_front_map = process_data(self.dff, self.velocities, filtered_map)

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

                if variance[h, j] > beta * 0.25 and (
                        type != 'cortex' or brain_mask[h, j] != 0):  # and spatial_coherence>0.5:
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

                    self.waveness[h, j, 3] = ratios[h, j]  # * wave_front_map[h, j]  # * spatial_coherence

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
        brain_mask = np.load('brain_mask_64.npy')
        brain_mask = brain_mask[:, :]
        outer_line_rgb = np.load('outer_line_rgb_64.npy')

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
            # ax.imshow(self.dff[:, :, i]*brain_mask[::-1,:32], cmap=phase_cmap(), vmin=-np.pi, vmax=np.pi,origin="lower")
            ax.imshow(self.dff[:, :, i] * brain_mask[::-1, :32], cmap=self.color_map, vmin=0, vmax=1, origin="lower")

            # ax.imshow(outer_line_rgba[:,:64])
            layout = outer_line_rgb[::-1, :32].astype(float)

            # If values are 0-255, normalize to 0-1
            if layout.max() > 1:
                layout = layout / 255.0

            # alpha = 1 where layout is 1, alpha = 0 elsewhere
            alpha_mask = (layout == 1).astype(float)

            # black RGB image
            black_layout = np.zeros((*layout.shape, 3))

            # ax.imshow(black_layout[:,:], alpha=alpha_mask)

            self.flows[i][:, :, 0][brain_mask[::-1, :32] == 0] = 0
            self.flows[i][:, :, 1][brain_mask[::-1, :32] == 0] = 0

            plot_quiver(ax, self.flows[i], spacing=6, scale=0.2, color='black')
            ax.set_ylim(self.dff.shape[0], 0)

            # ax.imshow(self.dff[:, :, i], cmap=phase_cmap(), vmin=-np.pi, vmax=np.pi)

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
        # image1 = ax.imshow(self.dff[:,:,0], cmap=phase_cmap(), vmin=-np.pi, vmax=np.pi)

        cbar = plt.colorbar(image1, ax=ax)
        # cbar.set_ticks([0, 1])
        # cbar.set_ticklabels(['0', '1'])

        # plt.figure(figsize=(2.5, 2.5))  # bigger figure in inches
        # plt.imshow(data.dff[:, :64, 80]*1.2, self.color_map, vmin=0, vmax=1)  # keep pixelated look
        # plt.axis('off')  # remove axes
        # plt.savefig("Barrel Activation.png", dpi=300, bbox_inches='tight')
        # plt.show()

        # plt.close()

        ani.save(f"{self.N}x{self.M}x{self.frames} {self.title} .mp4", writer='ffmpeg', fps=12.5)

        # frame = self.dff[:, :, 123]

        # Create a figure with desired size
        # fig, ax = plt.subplots(figsize=(8, 8))  # size in inches, adjust as needed
        # ax.imshow(frame, cmap=self.color_map)
        # ax.axis('off')  # remove axes

        # Save with higher DPI
        # fig.savefig("Tim_murphy_cortex.png", dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.close(fig)
        # plt.imsave("Tim_murphy_cortex.png", frame, cmap=self.color_map, dpi=300)

        plt.show()

    def full_analysis_3columns(self, space, scale, data_type):

        figsize_cm = (8, 4.5)
        figsize_in = tuple(x / 2.54 for x in figsize_cm)
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=figsize_in)  # 1 row, 4 columns
        data = self.data

        for axis in [ax, ax3]:
            axis.set_aspect('equal')

        outer_line_rgb = np.load('outer_line_rgb.npy')

        outer_line_rgba = np.ones((*outer_line_rgb.shape[:2], 4))  # Start with white and alpha = 1

        # Set the line color (dark gray) where needed
        line_mask = np.all(outer_line_rgb < 1.0, axis=2)  # Where the line is (i.e., not white)
        outer_line_rgba[..., :3][line_mask] = 0.392  # RGB to gray
        outer_line_rgba[..., 3] = 0.0  # Transparent everywhere
        outer_line_rgba[..., 3][line_mask] = 1.0  # Fully opaque where the line is

        self.sum_phase_space[self.sum_phase_space[:, :, -1, 0] == 0] = np.nan
        self.sum_phase_space[self.sum_phase_space[:, :, -1, 1] == 0] = np.nan

        if data_type == 'cortex':
            brain_mask = np.load('brain_mask.npy')
            outer_line_rgb = np.load('outer_line_rgb.npy')

            ax.imshow(outer_line_rgb)
            ax3.imshow(outer_line_rgb)
            ax.imshow(outer_line_rgba)

            plot_quiver(ax, self.sum_phase_space[:, :, -1, :], spacing=space, scale=scale, color='black', width=0.006)

            ax.set_ylim(self.dff.shape[0], 0)
            ax.set_xlim(0, self.dff.shape[1])
            ax.axis("off")

            flattened_values = data.waveness[:, :, 3][self.mask == 1]
            flattened_values = flattened_values.flatten()

            ax2.hist(flattened_values, bins=20, range=(0, 1), color='tomato', alpha=0.8, rwidth=0.9,
                     edgecolor='darkred')

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

        # data.waveness[:, :, 3] = np.where(data.waveness[:, :, 3] < lim, 0, data.waveness[:, :, 3])

        ax3.axis("off")
        cmap = plt.cm.hsv

        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 2 * np.pi))
        sm.set_array([])

        ax.set_aspect('equal')
        ax3.set_aspect('equal')

        fig2 = ax3.figure  # Get the figure that ax2 belongs to
        extent = ax3.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        # fig2.savefig("waveness_ax2_only_3121_3214.png", bbox_inches=extent, dpi=300, pad_inches=0)

        # plt.tight_layout()
        plt.savefig(f'{self.N}x{self.M}x{self.frames} {self.title}.pdf', dpi=400)
        # plt.savefig('Retina2.pdf')
        # plt.show()


colors = [
    (1.0, 1.0, 1.0),  # white
    (0.5, 0.7, 0.8),  # pale blue
    (0.1, 0.2, 0.6),  # navy blue
]

positions = [0.0, 0.5, 1]  # white at 0, pale blue at 0.2, navy at 0.5 (clipped early)

color_map = LinearSegmentedColormap.from_list("AbyssBlue", list(zip(positions, colors)))


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


# Example usage:
cmap = cyclic_hsv_cmap()

dff1 = np.load('Fig 2 Example.npy')
dff1_L = dff1

brain_mask = np.load('brain_mask_64.npy')
brain_mask = brain_mask[:, :32]

data = PreDataProcessing(dff1_L)
data.resize(64, 64)
data.dff = data.dff[:, :32, :]

data = FlowAnalyze(data)
data.dff = gaussian_filter(data.dff, sigma=[1.5, 1.5, 0])

dff1_L = data.dff

data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase=False)

from scipy.ndimage import label


def classify_jacobian_patterns(flow, dff1,
                               plane_threshold=0.8,
                               min_radius=4,
                               min_duration=2,
                               Nv=8,
                               alpha=1.2,
                               beta=0.3):
    """
    Pattern indices:
    0: Plane Wave
    1: Source
    2: Sink
    3: Saddle
    4: Standing Wave
    """
    N, M, T, _ = flow.shape
    delta_1 = alpha * 2 * np.pi / Nv * 3
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

    ### Standing-wave

    avg_mags = []

    for t in range(T):
        u_m = flow[:, :, t, 0][brain_mask]
        v_m = flow[:, :, t, 1][brain_mask]
        mags = np.sqrt(u_m ** 2 + v_m ** 2)
        if len(mags) > 0:
            avg_mags.append(np.mean(mags))

    avg_mags = np.array(avg_mags)
    standing_thresh = np.mean(avg_mags) - 2 * np.std(avg_mags)

    ### Spatial detection

    for t in range(T):

        u = flow[:, :, t, 0]
        v = flow[:, :, t, 1]

        u_m = u[brain_mask]
        v_m = v[brain_mask]

        mags = np.sqrt(u_m ** 2 + v_m ** 2)
        v_avg = np.mean(mags) if len(mags) > 0 else 0

        ### Standing wave (mutually exclusive with plane)

        if v_avg < standing_thresh:
            pattern_presence[4, t] = 1
            continue

        else:
            ### Plane wave (homogeneity R)

            if v_avg > 1e-5:
                sum_vec = np.array([np.sum(u_m), np.sum(v_m)])
                R = np.linalg.norm(sum_vec) / (mags.size * v_avg)
                print(R)
                # print(v_avg, np.round(R,4))

                if R >= plane_threshold:
                    pattern_presence[0, t] = 1

        ### Singularities (sources/sinks/saddles)

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

                if ptype == "Saddle":
                    raw_detections[t].append({'type': ptype, 'pos': np.array([cy, cx])})

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


def plot_pattern_raster(pattern_presence, fs=12.5):
    pattern_names = ["Plane", "Source", "Sink", "Saddle", "Standing"]
    pattern_presence[0, :] = (pattern_presence[0, :] == 2).astype(int)

    T = pattern_presence.shape[1]
    time = np.arange(T) / fs

    figsize_cm = (9, 4.5)
    figsize_in = tuple(x / 2.54 for x in figsize_cm)

    fig, ax = plt.subplots(figsize=figsize_in)

    # cmap = plt.get_cmap("BrBG") # Phase
    cmap = plt.get_cmap("PuOr_r")  # Intensity

    # example frame range
    frame_start = 1521 - (750 * 1 + 270)
    frame_end = 1592 - (750 * 1 + 270)

    # convert to seconds
    t_start = frame_start / fs
    t_end = frame_end / fs

    # highlight region
    ax.axvspan(t_start, t_end, color='tomato', alpha=0.2, zorder=100)

    # Truncate colormap from middle (0.5) to end (1.0)
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
            cmap(np.linspace(minval, maxval, n))
        )
        return new_cmap

    trunc_cmap = truncate_colormap(cmap, 0.5, 1)  # middle to end

    im = ax.imshow(
        pattern_presence,
        aspect='auto',
        origin='lower',
        extent=[0, time[-1], -0.5, 4.5],  # <-- FIXED
        cmap=trunc_cmap,
        rasterized=True,  # <-- add this
        interpolation="nearest",  # important
        resample=False  # important
    )

    ax.set_yticks(range(5))
    ax.set_yticklabels(pattern_names)
    ax.set_xticks([0, 1, 2, 3])

    ax.set_xlabel("Time (s)")
    ax.set_title("Example detection")
    # ax.set_xlim([0,120])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of detections")
    cbar.set_ticks([0, 4, 8])  # force min and max to appear

    plane_frames = np.where(pattern_presence[0, :] > 0)[0]
    n_plane_frames = len(plane_frames)
    plane_duration_sec = n_plane_frames / fs

    print("Number of Plane-wave frames:", n_plane_frames)
    print("Plane-wave duration (s):", plane_duration_sec)

    plt.tight_layout()
    plt.savefig("Raster_Detection.svg", bbox_inches='tight', transparent=False, dpi=300)
    plt.show()


def label_waves(pattern_map):
    labeled_waves = np.zeros_like(pattern_map)
    for t in range(pattern_map.shape[2]):
        labels, num = label(pattern_map[:, :, t])
        labeled_waves[:, :, t] = labels
    return labeled_waves


def animate_waves(data, flow, raw_detections, pattern_presence, interval=100):
    T = len(raw_detections)

    outer_line_rgb = np.load('outer_line_rgb_64.npy')
    outer_line_rgb = outer_line_rgb[:, 32:]

    brain_mask = np.load('brain_mask_64.npy')
    brain_mask = brain_mask[:, 32:]
    brain_mask = np.flipud(brain_mask)  # flip vertically

    fig, ax = plt.subplots(figsize=(6, 6))

    # im = ax.imshow(data[:, :, 0], cmap=color_map, origin="lower" ,vmin=0,vmax=1)
    im = ax.imshow(data[:, :, 0], cmap=cyclic_hsv_cmap(), origin="lower", vmin=-np.pi, vmax=np.pi)

    fig.colorbar(im, ax=ax)

    scat = ax.scatter([], [], s=100, edgecolor='white', linewidth=1.5)

    pattern_colors = {
        "Source": "red",
        "Sink": "blue",
        "Saddle": "green"
    }

    def update(frame):
        ax.cla()

        ax.imshow(data[:, :, frame], cmap=cyclic_hsv_cmap(), vmin=-np.pi, vmax=np.pi)
        # data[:, :, frame][brain_mask == 0] = 0

        # ax.imshow(data[:, :, frame], cmap=color_map, vmin=0, vmax=1, alpha=1)
        im.set_data(data[:, :, frame])
        flow[:, :, frame, 0][brain_mask == 0] = 0
        flow[:, :, frame, 1][brain_mask == 0] = 0

        plot_quiver(ax, flow[:, :, frame, :], spacing=2, scale=0.4, color='black')

        xs, ys, colors = [], [], []

        for det in raw_detections[frame]:
            y, x = det['pos']
            ptype = det['type']

            if ptype not in pattern_colors:
                continue

            xs.append(x)
            ys.append(y)
            colors.append(pattern_colors[ptype])

            ax.text(
                x + 2, y + 2,  # small offset so it doesn't overlap dot
                ptype,
                color=pattern_colors[ptype],
                fontsize=8,
                weight='bold',
                ha='left',
                va='bottom',
                bbox=dict(
                    facecolor='white',
                    alpha=0.6,
                    edgecolor='none',
                    pad=1
                )
            )

        # Scatter points
        if len(xs) > 0:
            ax.scatter(xs, ys,
                       s=100,
                       edgecolor='white',
                       linewidth=1.5,
                       c=colors)

        # Frame title
        title = f"Frame {frame}"

        if pattern_presence[0, frame] > 0:
            title += " | Plane Wave"
        if pattern_presence[4, frame] > 0:
            title += " | Standing Wave"

        overlay = outer_line_rgb[::-1].astype(np.float32)
        ax.imshow(overlay, alpha=(overlay > 0).astype(np.float32), cmap='Greys')

        ax.set_title(title)

        ax.set_ylim(data.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)

    plt.tight_layout()
    anim.save(f"1 Gaussian analysis .gif", writer='ffmpeg', fps=12.5)

    plt.show()

    return anim


flow_left = data.velocities.copy()

frame_labels_left, raw_detections_left = classify_jacobian_patterns(flow_left, dff1_L)

# print (raw_detections)

# plot_pattern_raster(frame_labels_R, fs=12.5)

# animate_waves(data.dff,data.velocities, raw_detections, frame_labels_R, interval=100)


dff1_R = dff1

brain_mask = np.load('brain_mask_64.npy')
brain_mask = brain_mask[:, 32:]

data = PreDataProcessing(dff1_R)
data.resize(64, 64)
data.dff = data.dff[:, 32:, :]
data = FlowAnalyze(data)
data.dff = gaussian_filter(data.dff, sigma=[1.5, 1.5, 0])

dff1_R = data.dff

data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase=False)

flow_right = data.velocities.copy()

frame_labels_right, raw_detections_right = classify_jacobian_patterns(flow_right, dff1_R)


def save_all_frames_figure(data, flow, raw_detections, pattern_presence, frames=None, ncols=8,
                           filename="all_frames.svg"):
    T = len(raw_detections)

    if frames is None:
        frames = range(T)

    frames = list(frames)
    n_frames = len(frames)
    nrows = math.ceil(n_frames / ncols)

    outer_line_rgb = np.load('outer_line_rgb_64.npy')
    outer_line_rgb = outer_line_rgb[:, 32:]

    brain_mask = np.load('brain_mask_64.npy')
    brain_mask = brain_mask[:, 32:]
    brain_mask = np.flipud(brain_mask)

    pattern_colors = {
        "Source": "red",
        "Sink": "blue",
        "Saddle": "green"
    }

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2.2, nrows * 2.2),
        squeeze=False
    )

    for ax, frame in zip(axes.flat, frames):

        frame_data = data[:, :, frame].copy()
        frame_data[brain_mask == 0] = 0

        ax.imshow(frame_data, cmap=color_map, vmin=0, vmax=1)

        flow_frame = flow[:, :, frame, :].copy()
        flow_frame[brain_mask == 0] = 0

        plot_quiver(ax, flow_frame, spacing=4, scale=0.08, color='black', width=0.01)

        xs, ys, colors = [], [], []

        for det in raw_detections[frame]:
            y, x = det["pos"]
            ptype = det["type"]

            if ptype not in pattern_colors:
                continue

            xs.append(x)
            ys.append(y)
            colors.append(pattern_colors[ptype])

            ax.text(
                x + 1,
                y + 1,
                ptype,
                color=pattern_colors[ptype],
                fontsize=5,
                weight="bold",
                ha="left",
                va="bottom",
                bbox=dict(
                    facecolor="white",
                    alpha=0.6,
                    edgecolor="none",
                    pad=0.5
                )
            )

        if len(xs) > 0:
            ax.scatter(
                xs,
                ys,
                s=25,
                edgecolor="white",
                linewidth=0.7,
                c=colors
            )

        title = f"{frame}"

        if pattern_presence[0, frame] > 0:
            title += " | Plane"
        if pattern_presence[4, frame] > 0:
            title += " | Standing"

        ax.set_title(title, fontsize=7)

        ax.set_ylim(data.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])

    # remove empty panels
    for ax in axes.flat[n_frames:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def plot_both_hemispheres_frames_and_raster(
        data_left,
        flow_left,
        raw_detections_left,
        pattern_left,
        brain_mask_left,
        data_right,
        flow_right,
        raw_detections_right,
        pattern_right,
        brain_mask_right,
        frames,
        fs=12.5,
        filename="both_hemispheres_frames_raster.svg"
):
    assert len(frames) == 8, "Please provide exactly 8 frame numbers."

    H, W = data_left.shape[:2]
    assert data_right.shape[:2] == (H, W), "Left/right shapes must match."

    pattern_names = ["Plane", "Source", "Sink", "Saddle", "Standing"]

    pattern_colors = {
        "Source": "tomato",
        "Sink": "royalblue",
        "Saddle": "mediumseagreen"
    }

    # combine raster from both hemispheres
    pattern_total = pattern_left + pattern_right

    figsize_cm = (18, 12)
    figsize_in = tuple(x / 2.54 for x in figsize_cm)

    fig = plt.figure(figsize=figsize_in, dpi=300)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Source',
               markerfacecolor=pattern_colors["Source"], markeredgecolor='white',
               markeredgewidth=0.7, markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Sink',
               markerfacecolor=pattern_colors["Sink"], markeredgecolor='white',
               markeredgewidth=0.7, markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Saddle',
               markerfacecolor=pattern_colors["Saddle"], markeredgecolor='white',
               markeredgewidth=0.7, markersize=7),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fontsize=7,
        borderaxespad=0.2
    )

    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 0.75], hspace=0.22, wspace=0.05)

    def draw_detections(ax, detections, x_offset=0):
        for det in detections:
            y, x = det["pos"]
            ptype = det["type"]

            if ptype not in pattern_colors:
                continue

            ax.scatter(x + x_offset, y, s=22, c=pattern_colors[ptype], edgecolor="white", linewidth=0.7, zorder=10)

    for k, frame in enumerate(frames):
        row = k // 4
        col = k % 4
        ax = fig.add_subplot(gs[row, col])

        # left hemisphere
        left_img = data_left[:, :, frame].copy()
        left_img[brain_mask_left == 0] = 0

        left_flow = flow_left[:, :, frame, :].copy()
        left_flow[brain_mask_left == 0] = 0

        # right hemisphere
        right_img = data_right[:, :, frame].copy()
        right_img[brain_mask_right == 0] = 0

        right_flow = flow_right[:, :, frame, :].copy()
        right_flow[brain_mask_right == 0] = 0

        # stitch into one full cortex frame
        full_img = np.zeros((H, 2 * W))
        full_img[:, :W] = left_img
        full_img[:, W:] = right_img

        full_flow = np.zeros((H, 2 * W, 2))
        full_flow[:, :W, :] = left_flow
        full_flow[:, W:, :] = right_flow

        ax.imshow(full_img, cmap=color_map, vmin=0, vmax=1)
        plot_quiver(ax, full_flow, spacing=4, scale=0.08, color="black", width=0.005)
        # detections
        draw_detections(ax, raw_detections_left[frame], x_offset=0)
        draw_detections(ax, raw_detections_right[frame], x_offset=W)

        title = f""

        if pattern_total[0, frame] == 2:
            title += "Plane"

        if pattern_total[4, frame] > 0:
            title += "Standing"

        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_ylim(H, 0)

    subgs_bottom = gs[2, :].subgridspec(1, 8, wspace=0.0)
    ax_raster = fig.add_subplot(subgs_bottom[0, 2:6])

    raster = pattern_total.copy()
    raster[0, :] = (raster[0, :] == 2).astype(int)  # same convention as your code

    cmap = plt.get_cmap("PuOr_r")

    def truncate_colormap(cmap, minval=0.5, maxval=1.0, n=256):
        return mcolors.LinearSegmentedColormap.from_list(
            f"trunc_{cmap.name}",
            cmap(np.linspace(minval, maxval, n))
        )

    trunc_cmap = truncate_colormap(cmap)

    T = raster.shape[1]
    duration_sec = (T - 1) / fs

    im = ax_raster.imshow(raster, aspect="auto", origin="lower", extent=[0, duration_sec, -0.5, 4.5], cmap=trunc_cmap,
                          interpolation="nearest", rasterized=True)

    # Y axis = pattern names
    ax_raster.set_yticks(range(5))
    ax_raster.set_yticklabels(pattern_names, fontsize=7)

    # X axis = seconds
    ax_raster.set_xlim(0, duration_sec)
    ax_raster.set_xlabel("Time (s)", fontsize=8)

    ax_raster.set_xticks(np.arange(0, np.floor(duration_sec) + 1, 1))

    cbar = fig.colorbar(im, ax=ax_raster, fraction=0.02, pad=0.02)
    cbar.set_ticks([0, 4, 8])
    cbar.ax.tick_params(labelsize=6)

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()


frames = [2, 16, 21, 27, 41, 47, 54, 68]

brain_mask_full = np.load('brain_mask_64.npy')
brain_mask_left = np.flipud(brain_mask_full[:, :32])
brain_mask_right = np.flipud(brain_mask_full[:, 32:])

plot_both_hemispheres_frames_and_raster(
    data_left=dff1_L,
    flow_left=flow_left,
    raw_detections_left=raw_detections_left,
    pattern_left=frame_labels_left,
    brain_mask_left=brain_mask_left,

    data_right=dff1_R,
    flow_right=flow_right,
    raw_detections_right=raw_detections_right,
    pattern_right=frame_labels_right,
    brain_mask_right=brain_mask_right,

    frames=frames,
    fs=12.5,
    filename="both_hemispheres_frames_raster.svg"
)


