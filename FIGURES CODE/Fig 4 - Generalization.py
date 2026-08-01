import numpy as np
import cv2
import scipy
from PIL import Image
from scipy.signal import butter, lfilter
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import map_coordinates
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter, maximum_filter, label
from scipy.ndimage import convolve
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, hsv_to_rgb
from Algos.Display import plot_quiver
from Algos.Create_Patterns import create_patterns, create_gaussians, create_gaussians_moving
from Algos.Data_Processing import Filter, resize, decrease_frame_rate, normalize_data
from Algos.Horn_Schunck import horn_schunck , horn_schunck_phase


#### Algorithm parameters###
alpha = 0.3   ## Optic Flow Horn-Schunck alpha parameter
iterations = 2  ## Number of iterations
n = 3  ## neighborhood size (2n+1)x(2n+1) around each pixel
lim = 0.5  ## Threshold of Wavness values for final score
beta = 0.035

plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42

import os


class MP4ToDff:
    def __init__(self, mp4):
        self.mp4 = mp4
        self.avi = mp4
        self.dff = np

    def mp4_video_to_numpy_gray(self):
        try:
            # Open the MP4 video file
            cap = cv2.VideoCapture(self.avi)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.avi}")
            # Get the video properties
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create an empty array to store the grayscale video frames
            video_array_gray = np.zeros((num_frames, height, width), dtype=np.uint8)

            # Read each frame, convert to grayscale, and store it in the array
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                video_array_gray[i] = frame_gray
            # Release the video capture object
            cap.release()

            self.dff = np.transpose(video_array_gray, (1, 2, 0))
            self.N = self.dff.shape[0]
            self.M = self.dff.shape[1]

        except Exception as e:
            print(f"Error processing video: {e}")


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


        brain_mask = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/brain_mask.npy')
        if self.dff.shape[1] == 64:
            brain_mask = brain_mask[:, :64]

        valid_mask = variance > (beta * 0.25)


        if type == 'cortex':
            # filtered_map = np.where(valid_mask & (brain_mask > 0), variance, 0)
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
                axes[1, 2].set_xlabel("Index")
                axes[1, 2].set_ylim([-0.15, 0.15])

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

        #wave_front_map = process_data(self.dff, self.velocities, filtered_map)

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


                if variance[h, j] > beta * 0.25 and (type != 'cortex' or brain_mask[h, j] != 0):  # and spatial_coherence>0.5:
                    sum_vx = np.sum(vector_field[..., 0])
                    sum_vy = np.sum(vector_field[..., 1])

                    avg_vx = sum_vx / ((2 * self.n) ** 2)
                    avg_vy = sum_vy / ((2 * self.n) ** 2)

                    angles = np.arctan2(avg_vx, avg_vy)
                    color = mcolors.hsv_to_rgb([(angles + np.pi) / (2 * np.pi), 1, 1])  # Saturation and Value are 1

                    self.waveness[h, j, 0] = angles
                    self.waveness[h, j, 1] = color[1]
                    self.waveness[h, j, 2] = color[2]

                    self.waveness[h, j, 3] = ratios[h, j] #* wave_front_map[h, j]  # * spatial_coherence

                    self.mask[h, j] = 1


                else:
                    self.waveness[h, j, 3] = 0


        self.Waveness = True



def bandpass_filter_video(video, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    print(low, high)

    b, a = butter(order, [low, high], btype='band')

    # Apply filter along the last axis (time)
    filtered = filtfilt(b, a, video, axis=-1)

    return filtered

def figure_plot_real_data(space, scale):
    """
    Plot six real-data examples without area or duration histograms.

    row5 and row6 must be dictionaries with:
        dff:          3-D NumPy array
        image_path:   path to the reference image
        title:        row title
        scores:       event-level waviness scores
        crop_half:    optional bool, default False
        quiver_scale: optional value, default scale
    """

    figsize = tuple(np.array((18, 20.25)) / 2.54)
    fig, axes = plt.subplots(6, 5, figsize=figsize)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def hide_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def clean_hist_axis(ax):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)

    def set_box_aspect_from_limits(ax):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if y1 != y0:
            ax.set_aspect((x1 - x0) / (y1 - y0), adjustable='box')

    def ratio_over_lim(values, mask):
        denominator = np.count_nonzero(mask)
        return np.count_nonzero(values > lim) / denominator if denominator else 0

    def extract_scores(dataset, fps=60, min_area=-0.1, offset=3700):
        scores = []
        for video_index, video in enumerate(dataset):
            last_end = -1
            for interval, values in sorted(dataset[video].items()):
                start, end = np.asarray(interval) + offset * video_index
                valid = (
                    values['area'] > min_area
                    and start >= last_end
                    and (end - start) / fps < 20
                )
                if valid:
                    scores.append(values['ratio'])
                    last_end = end
        return scores

    def cyclic_hsv_cmap(n=256, rotation=0.25):
        hue = (np.linspace(0, 1, n + 1) + rotation) % 1
        saturation = 0.7 + 0.3 * np.sin(2 * np.pi * hue)
        value = 0.85 + 0.15 * np.cos(2 * np.pi * hue)
        hsv = np.stack((hue, saturation, value), axis=1)
        rgb = hsv_to_rgb(hsv[None]).squeeze()[:-1]
        cmap = LinearSegmentedColormap.from_list('cyclic_hsv', rgb)
        cmap.set_bad('white')
        return cmap

    def percentage_hist(ax, values):
        if values is None or len(values) == 0:
            ax.axis('off')
            return

        values = np.asarray(values)
        weights = np.full(values.size, 100 / values.size)
        ax.hist(
            values,
            bins=np.linspace(0, 1, 11),
            weights=weights,
            color='#4A7DDB',
            edgecolor='black',
            linewidth=0.5,
        )
        ax.set(xlim=(0, 1), ylim=(0, 100))
        clean_hist_axis(ax)
        set_box_aspect_from_limits(ax)

    def score_scatter(ax, scores, seed=4):
        """Scatter plot used in the third row instead of a histogram."""
        scores = np.asarray(scores)
        x = np.random.default_rng(seed).normal(0, 0.035, scores.size)

        ax.scatter(
            x,
            scores,
            s=15,
            color='#5f86d6',
            edgecolor='black',
            linewidth=0.5,
            zorder=3,
        )
        mean_score = scores.mean()
        ax.plot(
            [-0.12, 0.12],
            [mean_score, mean_score],
            color='black',
            linewidth=1.5,
        )

        ax.set(
            xlim=(-0.18, 0.18),
            ylim=(0, 1),
            xticks=[],
            yticks=[0, 0.5, 1],
            ylabel='Waviness',
        )
        ax.yaxis.label.set_size(8)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.6)
        ax.tick_params(axis='y', labelsize=8, width=1.5, length=6)

    def analyze(dff, crop_half=False):
        processed = PreDataProcessing(dff)
        processed.resize(128, 128)

        if crop_half:
            processed.dff = processed.dff[:, :64, :]

        analyzed = FlowAnalyze(processed)
        analyzed.horn_schunck_flow(
            alpha=alpha,
            num_iter=iterations,
            phase=False,
        )
        analyzed.calculate_waveness(type='retina')
        return analyzed

    def plot_row(row, dff, image_path, title, scores=None, crop_half=False, quiver_scale=None, score_plot='hist'):
        ax_image, ax_flow, ax_hist, ax_map, ax_score = axes[row]

        # Reference image
        ax_image.imshow(mpimg.imread(image_path))
        ax_image.axis('off')

        data = analyze(dff, crop_half=crop_half)

        # Momentum / flow field
        flow = data.sum_phase_space[:, :, -1, :]
        if crop_half:
            flow = np.pad(flow, ((0, 0), (32, 32), (0, 0)))

        ax_flow.set_title(title)
        plot_quiver(
            ax_flow,
            flow,
            spacing=space,
            scale=scale if quiver_scale is None else quiver_scale,
            color='black',
            width=0.005,
        )
        ax_flow.set_ylim(flow.shape[0], 0)
        ax_flow.set_aspect('equal', adjustable='box')
        hide_axis(ax_flow)

        # Pixel-level waviness histogram
        values = data.waveness[:, :, 3][data.mask == 1].ravel()
        ax_hist.hist(
            values,
            bins=10,
            range=(0, 1),
            color='#d6b8a8',
            edgecolor='black',
            linewidth=0.5,
        )
        ax_hist.axvline(lim, color='dimgrey', linewidth=1, linestyle='--')
        ax_hist.set(
            xlim=(0, 1),
            ylim=(0, len(values)),
            yticks=[0, len(values)],
            yticklabels=[0, 1],
        )
        clean_hist_axis(ax_hist)
        set_box_aspect_from_limits(ax_hist)

        print(f'{title}: {ratio_over_lim(values, data.mask):.4f}')

        # Direction map
        activity_mask = np.zeros_like(data.waveness)
        activity_mask[..., :3] = 100 / 255
        activity_mask[..., 3] = np.where(data.mask == 1, 0.1, 0)

        alpha_map = np.where(
            data.waveness[:, :, 3] >= lim,
            data.waveness[:, :, 3],
            0,
        )

        ax_map.imshow(activity_mask)
        ax_map.imshow(
            data.waveness[:, :, 0],
            cmap=cmap,
            vmin=-np.pi,
            vmax=np.pi,
            alpha=alpha_map,
        )
        ax_map.set_aspect('equal', adjustable='box')
        hide_axis(ax_map)

        # Event-level score plot
        if score_plot == 'scatter':
            score_scatter(ax_score, scores)
        else:
            percentage_hist(ax_score, scores)

    # ------------------------------------------------------------------
    # Colormap and score data
    # ------------------------------------------------------------------
    cmap = cyclic_hsv_cmap()

    murphy_scores = []
    murphy_pattern = ('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/murphy_all_data_mouse{}.npy')
    for mouse in range(1, 5):
        dataset = np.load(murphy_pattern.format(mouse), allow_pickle=True).item()
        murphy_scores.extend(extract_scores(dataset))

    whisker_scores = [0.0382, 0.0313, 0.0789, 0.0333, 0.082, 0.4, 0.0695, 0.0513, 0.061, 0.0735, 0.1974, 0.0, 0.0053, 0.001, 0.0022, 0.0021, 0.0671,
                     0.026, 0.0, 0.0894, 0.011, 0.009, 0.0386, 0.0681, 0.0516, 0.0383, 0.0389, 0.0188, 0.011, 0.0601, 0.0462, 0.0314, 0.0, 0.2353,
                    0.1142, 0.114, 0.1841, 0.7381, 0.145, 0.0879, 0.3125, 0.0595, 0.1029, 0.128, 0.1663, 0.126, 0.1369, 0.0, 0.0577, 0.1338, 0.0179,
                    0.082, 0.0984, 0.0053, 0.0043, 0.0927, 0.0387, 0.0646, 0.0631, 0.0067, 0.0291, 0.0571, 0.049, 0.0531, 0.0698, 0.1344, 0.0558,
                    0.0341, 0.0, 0.0588, 0.2083, 0.0421, 0.0379, 0.029, 0.0053, 0.067, 0.0168, 0.0061, 0.0003, 0.0011]

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    murphy = np.load('/Users/arielrom/Downloads/murphy_56to60.npy')[:, :, 12352:12729]

    whisker = np.load('whisker_stim_right/232FN_whis_right1_Pos0.npy')[:, :, 130:-50][:, :, 250:298]

    retina = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/retina_dF_F_normed.npy')

    #spreading_depression = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/SD_df_f_NORMED.npy')[35:163, 40:168, 250:-290]
    #spreading_depression = decrease_frame_rate(spreading_depression,20,10,)




    picture_dir = ('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/Pictures')

    plot_row(0, murphy, f'{picture_dir}/Tim_murphy_cortex.png', 'Tim Murphy Cortex', scores=murphy_scores, quiver_scale=2 * scale)

    plot_row(1, whisker, f'{picture_dir}/Barrel Activation.png', 'Whisker stim right', scores=whisker_scores, crop_half=True,)

    plot_row(2, spreading_depression, f'{picture_dir}/Spreading Depression.png', 'Spreading Depression' )

    retina_scores = np.array([0.96, 0.87, 0.83, 0.80, 0.77, 0.61])

    plot_row(3, retina, f'{picture_dir}/Retina.png', 'Retina', scores=retina_scores, score_plot='scatter')


    # Bottom-row labels only
    axes[5, 2].set_xlabel('Pixel waviness', fontsize=10)
    axes[5, 4].set_xlabel('Score', fontsize=10)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig('Real Examples.pdf', bbox_inches='tight', dpi=1000)
    plt.show()


figure_plot_real_data(space = 5, scale = 0.15)
