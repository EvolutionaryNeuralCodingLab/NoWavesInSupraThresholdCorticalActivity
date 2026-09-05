import numpy as np
import cv2
import scipy
import h5py
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.signal import butter, lfilter
import tifffile as tiff
from numpy.lib.stride_tricks import sliding_window_view
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

        area = round(activated_pixels / brain_activity_area, 2)

        def velocities():
            velocity_magnitudes = np.linalg.norm(self.velocities, axis=-1)  # shape: (N, M, frames-2)
            valid_magnitudes = velocity_magnitudes[valid_mask]  # 1D array of valid values
            valid_magnitudes = np.ravel(valid_magnitudes)

            # Define fixed bin edges between 0 and 5
            bin_edges = np.linspace(0, 0.5, 51)  # 50 bins → 51 edges

            # Histogram using fixed bin edges
            counts, bins = np.histogram(valid_magnitudes, bins=bin_edges)

            # Convert to percentages
            percentages = counts / counts.sum() * 100

            # Plot
            plt.bar(bins[:-1], percentages, width=np.diff(bins), align='edge',
                    color='mediumseagreen', edgecolor='black')

            # Set x-axis limits to match your fixed bin range
            plt.xlim(0, 0.5)

            plt.xlabel('Velocity Magnitude ')
            plt.ylabel('Percentage of Values (%)')
            plt.title(f'Radial Wave')  # , Relative area = {area}')
            plt.grid(True)
            plt.show()

        if type == 'cortex':
            # filtered_map = np.where(valid_mask & (brain_mask > 0), variance, 0)
            filtered_map = np.where(valid_mask, variance, 0)

        else:
            filtered_map = np.where(valid_mask, variance, 0)

        def compute_ftle(vel_x, vel_y, dt=1.0):

            def runge_kutta(vel_x, vel_y, X_final, Y_final, dt):
                """
                Runge-Kutta 4th order integration for velocity field.

                Parameters:
                    vel_x, vel_y (numpy array): The velocity fields (MxNxT)
                    X_final, Y_final (numpy array): The final positions to be updated
                    dt (float): Time step

                Returns:
                    X_final, Y_final (numpy array): Updated positions
                """
                # 1st stage: k1
                kx1 = vel_x * dt
                ky1 = vel_y * dt

                # 2nd stage: k2
                kx2 = (vel_x + 0.5 * kx1) * dt
                ky2 = (vel_y + 0.5 * ky1) * dt

                # 3rd stage: k3
                kx3 = (vel_x + 0.5 * kx2) * dt
                ky3 = (vel_y + 0.5 * ky2) * dt

                # 4th stage: k4
                kx4 = (vel_x + kx3) * dt
                ky4 = (vel_y + ky3) * dt

                # Update final positions based on Runge-Kutta method
                X_final += (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
                Y_final += (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6

                return X_final, Y_final

            """
            Computes the FTLE field for a given velocity field.

            Parameters:
                vel_x (numpy array): MxNxT array of x velocity components
                vel_y (numpy array): MxNxT array of y velocity components
                dt (float): Time step between frames

            Returns:
                numpy array: MxN array of FTLE values
            """

            print("Max velocity:", np.max(vel_x), np.max(vel_y))
            print("Min velocity:", np.min(vel_x), np.min(vel_y))

            M, N, T = vel_x.shape
            X, Y = np.meshgrid(np.arange(N), np.arange(M))
            X_final, Y_final = X.astype(np.float64), Y.astype(np.float64)  # Ensure float type

            # Integrate flow map over time using Euler method
            for t in range(T):
                X_final, Y_final = runge_kutta(vel_x[:, :, t], vel_y[:, :, t], X_final, Y_final, dt)

            # Compute Jacobian (Deformation Gradient)
            dXdx = np.gradient(X_final, axis=1)
            dXdy = np.gradient(X_final, axis=0)
            dYdx = np.gradient(Y_final, axis=1)
            dYdy = np.gradient(Y_final, axis=0)

            # Compute FTLE
            ftle_field = np.zeros((M, N))
            for i in range(M):
                for j in range(N):
                    F = np.array([[dXdx[i, j], dXdy[i, j]],
                                  [dYdx[i, j], dYdy[i, j]]])
                    C = F.T @ F  # Cauchy-Green strain tensor
                    eigvals = np.linalg.eigvals(C)
                    max_eigval = max(eigvals)
                    ftle_field[i, j] = np.log(np.sqrt(max_eigval)) / T

            return ftle_field

        # ftle_field = compute_ftle(self.velocities[:,:,:,0],self.velocities[:,:,:,1])
        # plt.imshow(returns_to_origin, cmap='viridis', interpolation='nearest',vmin=0,vmax=100)
        # plt.colorbar()
        # plt.title('Finite Time Lyapunov Exponent - Cortex 2')
        # plt.show()

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

        num_ones = np.sum(self.mask == 1)
        binary_map = (self.waveness[:, :, 3] > lim).astype(int)

        # labeled_array, num_features = ndimage.label(binary_map)
        # sizes = ndimage.sum(binary_map, labeled_array, range(1, num_features + 1))
        # weighted_avg_size = np.average(sizes, weights=sizes)

        # print(weighted_avg_size/  brain_activity_area)
        # print(area)
        # plt.imshow(wave_front_map, cmap='viridis', aspect='equal')
        # plt.colorbar(label='Value')
        # plt.title('Wave Front Map')
        # plt.show()

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
        self.alpha = alpha
        self.beta = beta
        self.iter = iterations
        self.lim = lim

    def plot_data(self):

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

        cmap = phase_cmap()
        cmap.set_bad(color="white")

        fig, ax = plt.subplots()
        # title = "test"
        # self.title="sda"
        brain_mask = np.load('brain_mask.npy')[:,:64]
        outer_line_rgb = np.load('outer_line_rgb.npy')
        outer_line_rgb = outer_line_rgb[:, :]
        outer_line_rgba = np.ones((*outer_line_rgb.shape[:2], 4))  # Start with white and alpha = 1
        line_mask = np.all(outer_line_rgb < 1.0, axis=2)  # Where the line is (i.e., not white)
        outer_line_rgba[..., :3][line_mask] = 0.392  # RGB to gray
        outer_line_rgba[..., 3] = 0.0  # Transparent everywhere
        outer_line_rgba[..., 3][line_mask] = 1.0  # Fully opaque where the line is

        binary_mask = (outer_line_rgb == 1)  # or 1, depending on your actual mask

        def anima(i):
            ax.cla()
            #self.dff[:, :, i][brain_mask[:,:] == 0] = np.nan

            #self.dff[:,:,i] *= brain_mask[:,:64]
            ax.imshow(self.dff[:, :, i], cmap=self.color_map, vmin=0, vmax=1)
            #ax.imshow(self.dff[:, :, i], cmap=cmap, vmin=-np.pi, vmax=np.pi)

            #ax.imshow(outer_line_rgba[:,:])
            #ax.imshow(outer_line_rgba)
            #plot_quiver(ax, self.phase_space[:,:,i,], spacing=4, scale=0.03, color='black')
            #plot_quiver(ax, self.flows[i], spacing=4, scale=0.04, color='black')
            #ax.set_ylim(self.dff.shape[0], 0)

            #ax.imshow(self.dff[:, :, i], cmap=cyclic_hsv_cmap(), vmin=-np.pi, vmax=np.pi)

            # Assume outer_line_rgb has shape (H, W, 3) and contains (0.392, 0.392, 0.392) where the line is

            # ax.set_title(fr"{self.N}x{self.M} {self.title} ")

            time_sec = i / 25
            ax.set_title(f"Time: {int(time_sec // 60)} min {int(time_sec % 60)} sec")
            #        transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white",
            #        bbox=dict(facecolor='black', alpha=0.5))

            # ax.axis('off')

            return ax

        ani = animation.FuncAnimation(fig, func=anima, frames=self.frames - 1, blit=False, interval=32)
        image1 = ax.imshow(self.dff[:, :, 0], cmap=self.color_map, vmin=0, vmax=1)

        #image1 = ax.imshow(self.dff[:, :, 0], cmap=cmap, vmin=-np.pi, vmax=np.pi)
        #image1 = ax.imshow(self.dff[:,:,0], cmap=cyclic_hsv_cmap(), vmin=-np.pi, vmax=np.pi)

        cbar = plt.colorbar(image1, ax=ax)
        # cbar.set_ticks([0, 1])
        #cbar.set_ticks([-np.pi, 0, np.pi])
        #cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])

        # plt.figure(figsize=(2.5, 2.5))  # bigger figure in inches
        # plt.imshow(data.dff[:, :64, 80]*1.2, self.color_map, vmin=0, vmax=1)  # keep pixelated look
        # plt.axis('off')  # remove axes
        # plt.savefig("Barrel Activation.png", dpi=300, bbox_inches='tight')
        # plt.show()

        # plt.close()

        #ani.save(f"{self.N}x{self.M}x{self.frames} {self.title} .mp4", writer='ffmpeg', fps=12.5)
        #ani.save(f"{self.title}.mp4", writer='ffmpeg', fps=10)


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

        self.sum_phase_space[:, :, -1, :]


        if data_type == 'cortex':
            brain_mask = np.load('brain_mask.npy')
            outer_line_rgb = np.load('outer_line_rgb.npy')

            ax.imshow(outer_line_rgb)
            ax3.imshow(outer_line_rgb)
        #ax.imshow(outer_line_rgba)

        plot_quiver(ax, self.sum_phase_space[:, :, -1, :], spacing=space, scale=scale, color='black', width=0.006)

        ax.set_ylim(self.dff.shape[0], 0)
        ax.set_xlim(0, self.dff.shape[1])
        ax.axis("off")

        flattened_values = data.waveness[:, :, 3][data.mask==1].flatten()

        ax2.hist(flattened_values, bins=10, range=(0, 1), color='#d6b8a8', alpha=1, rwidth=1, edgecolor='#000000',linewidth=0.5)
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

        #ax3.imshow(outer_line_rgba)
        norm = plt.Normalize(vmin=0, vmax=2 * np.pi)

        def cyclic_hsv_cmap(n=256):
            # Create n+1 points to ensure cyclic closure
            hue = (np.linspace(0, 1, n + 1) + 1 / 3) % 1

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
        #plt.savefig(f'{self.N}x{self.M}x{self.frames} {self.title}.pdf', dpi=400)
        # plt.savefig('Retina2.pdf')
        #plt.show()

#### FUNCTIONS TO CALCULATE WAVINESS SCORE IN DIFFERENT CONDITIONS

def compare_basic_patterns_velocities(noise, std, N=128, M=64, frames=65, x0=32, y0=70 - 18, sd0=16, t0_0=22 + 22.5, sdT0=10, x1=32, y1=70 + 18, sd1=16, t0_1=22, sdT1=10):
    def ratio_over_lim(lim, flattened_values, mask_activity):
        total_above_lim = np.count_nonzero(flattened_values[flattened_values > lim])
        total_non_zero = np.count_nonzero(mask_activity)

        if total_non_zero > 0:
            ratio = total_above_lim / total_non_zero
        else:
            ratio = 0  # or handle the division by zero case as you prefer

        return ratio

    fps_values = [2.5, 5, 10, 20, 40]

    brain_mask = np.load('brain_mask.npy')
    brain_mask = brain_mask[:, :64]

    binary_mask = (brain_mask == 1)  # or 1, depending on your actual mask

    x, y = np.meshgrid(np.linspace(0, M - 1, M), np.linspace(0, N - 1, N))

    dff_40fps_2gaussian = np.zeros((N, M, 4 * frames))
    for t in range(4 * frames):
        gaussian1 = np.exp(- (t / 4 - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                    * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

        gaussian2 = np.exp(- (t / 4 - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                    * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))

        dff_40fps_2gaussian[:, :, t] = gaussian1 + gaussian2
    dff_1_4_plane = create_patterns(N=N, M=M, frames=frames, pattern='plane', x0=32, y0=90, sd0=8, u=0, v=0.25,
                                    rad_spd=0.25, rad_width=5)
    dff_1_4_radial = create_patterns(N=N, M=M, frames=frames, pattern='radial', x0=M / 2, y0=N / 2, sd0=8, u=0, v=0.25,
                                     rad_spd=0.25, rad_width=5)

    dff_20fps_2gaussian = np.zeros((N, M, 2 * frames))
    for t in range(2 * frames):
        gaussian1 = np.exp(- (t / 2 - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                    * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

        gaussian2 = np.exp(- (t / 2 - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                    * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))

        dff_20fps_2gaussian[:, :, t] = gaussian1 + gaussian2
    dff_1_2_plane = create_patterns(N=N, M=M, frames=frames, pattern='plane', x0=32, y0=90, sd0=8, u=0, v=0.5,
                                    rad_spd=0.5, rad_width=5)
    dff_1_2_radial = create_patterns(N=N, M=M, frames=frames, pattern='radial', x0=M / 2, y0=N / 2, sd0=8, u=0, v=0.5,
                                     rad_spd=0.5, rad_width=5)

    dff_10fps_2gaussian = np.zeros((N, M, frames))
    for t in range(frames):
        gaussian1 = np.exp(- (t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                    * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

        gaussian2 = np.exp(- (t - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                    * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))

        dff_10fps_2gaussian[:, :, t] = gaussian1 + gaussian2
    dff_1_plane = create_patterns(N=N, M=M, frames=frames, pattern='plane', x0=32, y0=90, sd0=8, u=0, v=1,
                                  rad_spd=1, rad_width=5)
    dff_1_radial = create_patterns(N=N, M=M, frames=frames, pattern='radial', x0=M / 2, y0=N / 2, sd0=8, u=0, v=1,
                                   rad_spd=1, rad_width=5)

    dff_5fps_2gaussian = np.zeros((N, M, int(frames / 2)))
    for t in range(int((frames / 2))):
        gaussian1 = np.exp(- (2 * t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                    * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

        gaussian2 = np.exp(- (2 * t - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                    * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))

        dff_5fps_2gaussian[:, :, t] = gaussian1 + gaussian2
    dff_2_plane = create_patterns(N=N, M=M, frames=frames, pattern='plane', x0=32, y0=90, sd0=8, u=0, v=2 * 1,
                                  rad_spd=2 * 1, rad_width=5)
    dff_2_radial = create_patterns(N=N, M=M, frames=frames, pattern='radial', x0=M / 2, y0=N / 2, sd0=8, u=0, v=2 * 1,
                                   rad_spd=2 * 1, rad_width=5)

    dff_2_5fps_2gaussian = np.zeros((N, M, int(frames / 4)))
    for t in range(int((frames / 4))):
        gaussian1 = np.exp(- (4 * t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                    * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

        gaussian2 = np.exp(- (4 * t - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                    * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))

        dff_2_5fps_2gaussian[:, :, t] = gaussian1 + gaussian2
    dff_4_plane = create_patterns(N=N, M=M, frames=frames, pattern='plane', x0=32, y0=90, sd0=8, u=0, v=4 * 1,
                                  rad_spd=4 * 1, rad_width=5
                                  )
    dff_4_radial = create_patterns(N=N, M=M, frames=frames, pattern='radial', x0=M / 2, y0=N / 2, sd0=8, u=0, v=4 * 1,
                                   rad_spd=4 * 1, rad_width=5)

    def stats():
        variance = np.var(data.dff, axis=2)

        plt.figure(figsize=(8, 6))
        plt.imshow(variance, cmap='viridis')  # you can try other colormaps too
        plt.colorbar(label='Value')
        plt.title('My Map')
        plt.axis('off')  # or plt.axis('on') if you want the axes
        plt.show()

        N, M, T = data.dff.shape

        # Compute std per pixel across time
        std_map = np.std(data.dff, axis=2)
        max_map = np.max(data.dff, axis=2)

        # Plot side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # STD map
        im1 = axs[0].imshow(std_map, cmap='hot')
        axs[0].set_title('Pixel-wise STD Over Time')
        axs[0].axis('off')
        fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

        # MAX map
        im2 = axs[1].imshow(max_map, cmap='viridis')
        axs[1].set_title('Pixel-wise MAX Over Time')
        axs[1].axis('off')
        fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    ratios = []
    ratios_2gaussian = {}
    dff_dict = {
        2.5: dff_2_5fps_2gaussian,
        5: dff_5fps_2gaussian,
        10: dff_10fps_2gaussian,
        20: dff_20fps_2gaussian,
        40: dff_40fps_2gaussian
    }
    for fps in fps_values:
        dff = dff_dict[fps]
        data = PreDataProcessing(dff)
        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=1)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask

        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)
        data.calculate_waveness(type='cortex')

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_2gaussian[fps] = ratio
        ratios.append(np.round(ratio, 4))

        display = Display(data)
        display.title = f'2gaussians velocity = {fps} , noise = {std}'
        #display.plot_data()
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")
    print('2 gaussian: ', ratios)

    ratios = []
    ratios_plane = {}
    dff_dict = {
        2.5: dff_1_4_plane,
        5: dff_1_2_plane,
        10: dff_1_plane,
        20: dff_2_plane,
        40: dff_4_plane
    }
    for fps in fps_values:
        dff = dff_dict[fps]
        data = PreDataProcessing(dff)
        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=1)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask
        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)
        data.calculate_waveness(type='cortex')

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_plane[fps] = ratio
        ratios.append(np.round(ratio, 4))
        display = Display(data)
        display.title = f'plane velocity = {fps} , noise = {std}'
        #display.plot_data()
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")
    print('plane: ', ratios)

    ratios = []
    ratios_radial = {}
    dff_dict = {
        2.5: dff_1_4_radial,
        5: dff_1_2_radial,
        10: dff_1_radial,
        20: dff_2_radial,
        40: dff_4_radial
    }
    for fps in fps_values:
        dff = dff_dict[fps]
        data = PreDataProcessing(dff)
        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=1)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask

        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)
        data.calculate_waveness(type='cortex')

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_radial[fps] = ratio
        ratios.append(np.round(ratio, 4))
        display = Display(data)
        display.title = f'radial velocity = {fps} , noise = {std}'
        #display.plot_data()
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")
    print('radial: ', ratios)

def compare_basic_patterns_intensities(noise, std, N=128, M=64, frames=65, x0=32, y0=70 - 18, sd0=16, t0_0=22 + 22.5,sdT0=10, x1=32, y1=70 + 18, sd1=16, t0_1=22, sdT1=10):
    intensities_values = [0.2, 0.4, 0.6, 0.8, 1]

    def ratio_over_lim(lim, flattened_values, mask_activity):
        total_above_lim = np.count_nonzero(flattened_values[flattened_values > lim])
        total_non_zero = np.count_nonzero(mask_activity)

        if total_non_zero > 0:
            ratio = total_above_lim / total_non_zero
        else:
            ratio = 0  # or handle the division by zero case as you prefer

        return ratio

    brain_mask = np.load('brain_mask.npy')
    brain_mask = brain_mask[:, :64]

    binary_mask = (brain_mask == 1)  # or 1, depending on your actual mask

    x, y = np.meshgrid(np.linspace(0, M - 1, M), np.linspace(0, N - 1, N))

    ratios_2gaussian = []
    for I in intensities_values:
        dff_10fps_2gaussian = np.zeros((N, M, frames))
        for t in range(frames):
            gaussian1 = np.exp(- (t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                        * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

            gaussian2 = np.exp(- (t - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                        * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))

            dff_10fps_2gaussian[:, :, t] = gaussian1 + gaussian2
        dff_10fps_2gaussian = I * dff_10fps_2gaussian
        data = PreDataProcessing(dff_10fps_2gaussian)

        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=I)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask

        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'2 gaussians , max amp = {I} , std={std}'
        #display.plot_data()

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_2gaussian.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("gaussians:", ratios_2gaussian)

    ratios_plane = []
    for I in intensities_values:
        dff = create_patterns(N=N, M=M, frames=frames, pattern='plane', x0=32, y0=90, sd0=8, u=0, v=1,
                              rad_spd=1, rad_width=5)
        dff = I * dff

        data = PreDataProcessing(dff)
        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=I)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask
        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'plane , max amp = {I}'
        # display.plot_data()

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_plane.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("plane:", ratios_plane)

    ratios_radial = []
    for I in intensities_values:
        dff = create_patterns(N=N, M=M, frames=frames, pattern='radial', x0=M / 2, y0=N / 2, sd0=8, u=0, v=1,
                              rad_spd=1, rad_width=5)
        dff = I * dff
        data = PreDataProcessing(dff)
        if noise == True:
            data.add_noise(std)

            data.dff = np.clip(data.dff, a_min=0, a_max=I)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask

        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'radial , max amp = {I}'
        # display.plot_data()

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_radial.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("radial:", ratios_radial)

def compare_wave_gap(noise, std, N=128, M=64, frames=65, x0=32, y0=70 - 18, sd0=16, t0_0=22 + 22.5, sdT0=10, x1=32,y1=70 + 18, sd1=16, t0_1=22, sdT1=10):
    gap_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def ratio_over_lim(lim, flattened_values, mask_activity):
        total_above_lim = np.count_nonzero(flattened_values[flattened_values > lim])
        total_non_zero = np.count_nonzero(mask_activity)

        if total_non_zero > 0:
            ratio = total_above_lim / total_non_zero
        else:
            ratio = 0  # or handle the division by zero case as you prefer

        return ratio

    brain_mask = np.load('brain_mask.npy')
    brain_mask = brain_mask[:, :64]

    binary_mask = (brain_mask == 1)  # or 1, depending on your actual mask

    x, y = np.meshgrid(np.linspace(0, M - 1, M), np.linspace(0, N - 1, N))

    def typical_grad(dff):
        active_mask_x = dff[:-1, :, :] > 0.15  # shape matches grad_x
        active_mask_y = dff[:, :-1, :] > 0.15  # shape matches grad_y
        active_mask_t = dff[:, :, :-1] > 0.15  # shape matches grad_t

        # Select only active gradients
        grad_x_active = np.diff(dff, axis=0)[active_mask_x]  # grad_x
        grad_y_active = np.diff(dff, axis=1)[active_mask_y]  # grad_y
        grad_t_active = np.diff(dff, axis=2)[active_mask_t]  # grad_t

        # Combine
        all_grads = np.concatenate([grad_x_active.ravel(), grad_y_active.ravel(), grad_t_active.ravel()])
        typical_grad = np.mean(np.abs(all_grads))  # or np.median for robustness

        return typical_grad

    dff_10fps_2gaussian = np.zeros((N, M, frames))
    for t in range(frames):
        gaussian1 = np.exp(- (t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                    * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

        gaussian2 = np.exp(- (t - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                    * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))

        dff_10fps_2gaussian[:, :, t] = gaussian1 + gaussian2
    # typical_grad = typical_grad(dff_10fps_2gaussian)

    ratios_2gaussian = []
    for gap in gap_values:
        dff_10fps_2gaussian = np.zeros((N, M, frames))
        for t in range(frames):
            gaussian1 = np.exp(- (t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                        * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

            gaussian2 = np.exp(- (t - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                        * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))

            dff_10fps_2gaussian[:, :, t] = gaussian1 + gaussian2

        dff_10fps_2gaussian[80:80 + gap, 0:63, :] = 0
        data = PreDataProcessing(dff_10fps_2gaussian)

        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=1)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask

        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'2 gaussian gap={gap}, noise = {std}'
        #display.plot_data()

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_2gaussian.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("gaussians:", ratios_2gaussian)


    ratios_plane = []
    for gap in gap_values:
        dff = create_patterns(N=N, M=M, frames=frames, pattern='plane', x0=32, y0=90, sd0=8, u=0, v=1,rad_spd=1, rad_width=5)
        dff[67:67 + gap, 0:63, :] = 0

        data = PreDataProcessing(dff)
        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=1)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask
        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'plane gap={gap}, noise = {std}'
        #display.plot_data()

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_plane.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("plane:", ratios_plane)


    ratios_radial = []
    for gap in gap_values:
        dff = create_patterns(N=N, M=M, frames=frames, pattern='radial', x0=M / 2, y0=N / 2, sd0=8, u=0, v=1,
                              rad_spd=1, rad_width=5)
        dff[67:67 + gap, 0:63, :] = 0

        data = PreDataProcessing(dff)
        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=1)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask

        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'radial gap={gap}, noise = {std}'
        #display.plot_data()

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_radial.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("radial:", ratios_radial)


### WIDTH FOR WAVES AND GAUSSIANS

def compare_waves_width(noise, std, N=128, M=64):
    width_values = [2,4,6,8,10, 12, 14, 16, 18,20]

    def ratio_over_lim(lim, flattened_values, mask_activity):
        total_above_lim = np.count_nonzero(flattened_values[flattened_values > lim])
        total_non_zero = np.count_nonzero(mask_activity)

        if total_non_zero > 0:
            ratio = total_above_lim / total_non_zero
        else:
            ratio = 0  # or handle the division by zero case as you prefer

        return ratio

    brain_mask = np.load('brain_mask.npy')
    brain_mask = brain_mask[:, :64]

    binary_mask = (brain_mask == 1)  # or 1, depending on your actual mask

    x, y = np.meshgrid(np.linspace(0, M - 1, M), np.linspace(0, N - 1, N))

    ratios_plane = []
    for width in width_values:
        dff = create_patterns(N=N, M=M, frames=125, pattern='plane', x0=32, y0=128, sd0=8, u=0, v=1, rad_spd=1,rad_width=width)

        data = PreDataProcessing(dff)
        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=1)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask
        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'plane, FWHM = {np.round(width*2.355*(9/128),2)}'
        #display.plot_data()

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_plane.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("plane:", ratios_plane)

    ratios_radial = []
    for width in width_values:
        dff = create_patterns(N=N, M=M, frames=65, pattern='radial', x0=M / 2, y0=N / 2, sd0=8, u=0, v=1, rad_spd=1,rad_width=width)

        data = PreDataProcessing(dff)
        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=1)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask

        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'radial, FWHM = {np.round(width*2.355*(9/128),2)}'
        #display.plot_data()

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_radial.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("radial:", ratios_radial)

def compare_gaussians_width(noise, std, N=128, M=64, frames=65, x0=32, y0=70, sd0=16, t0_0=22 + 22.5, sdT0=10, x1=32,y1=70 + 20, sd1=16, t0_1=22, sdT1=10):
    sigma_spatial = [2,4,6, 8, 10, 12, 14, 16, 18]

    def ratio_over_lim(lim, flattened_values, mask_activity):
        total_above_lim = np.count_nonzero(flattened_values[flattened_values > lim])
        total_non_zero = np.count_nonzero(mask_activity)

        if total_non_zero > 0:
            ratio = total_above_lim / total_non_zero
        else:
            ratio = 0  # or handle the division by zero case as you prefer

        return ratio

    brain_mask = np.load('brain_mask.npy')
    brain_mask = brain_mask[:, :64]

    binary_mask = (brain_mask == 1)  # or 1, depending on your actual mask

    x, y = np.meshgrid(np.linspace(0, M - 1, M), np.linspace(0, N - 1, N))

    ratios_2gaussian = []
    for sigma in sigma_spatial:
        y0 = 70 - 1.125 * sigma
        sd0 = sigma
        y1 = 70 + 1.125 * sigma
        sd1 = sigma

        dff_10fps_2gaussian = np.zeros((N, M, frames))
        for t in range(frames):
            gaussian1 = np.exp(- (t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                        * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

            gaussian2 = np.exp(- (t - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                        * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))

            dff_10fps_2gaussian[:, :, t] = gaussian1 + gaussian2
        data = PreDataProcessing(dff_10fps_2gaussian)

        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=1)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask

        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations , phase= False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'2 gaussians ,FWHM = {np.round(sigma*2.355*(9/128),2)}'
        #display.plot_data()

        flattened_values = data.waveness[:, :, 3][data.mask == 1]
        flattened_values = flattened_values.flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_2gaussian.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("gaussians:", ratios_2gaussian)

#compare_basic_patterns_velocities(noise = True, std=0.03)
#compare_basic_patterns_intensities(noise = True, std=0.03)
#compare_wave_gap(noise = True, std=0.03)
#compare_waves_width(noise = True, std=0.03)
#compare_gaussians_width(noise=True, std=0.03)





def diff_conditions_comparison():
    ### Velocities ###
    gaussian_vel_015 = [0.027, 0.051, 0.077, 0.025, 0.004]
    plane_vel_015 = [0.997, 1.000, 1.000, 0.997, 0.986]
    radial_vel_015 = [0.955, 0.988, 0.991, 0.981, 0.964]

    gaussian_vel_03 = [0.024, 0.047, 0.022, 0.000, 0.000]
    plane_vel_03 = [0.829, 1.000, 0.999, 0.997, 0.970]
    radial_vel_03 = [0.649, 0.982, 0.987, 0.984, 0.948]

    gaussian_vel_06 = [0.034, 0.026, 0.002, 0.000, 0.000]
    plane_vel_06 = [0.122, 0.794, 0.985, 0.989, 0.939]
    radial_vel_06 = [0.002, 0.636, 0.973, 0.978, 0.928]

    figsize_cm = (18, 15)  # example in cm
    figsize_in = tuple(x / 2.54 for x in figsize_cm)  # convert to inches

    fig, axs = plt.subplots(3, 4, figsize=figsize_in, sharex='col')

    x_axis = np.array([1, 2, 3, 4, 5])

    axs[0, 0].plot(x_axis, gaussian_vel_015, label='2 Gaussians', color="mediumseagreen", linewidth=2)
    axs[0, 0].plot(x_axis, plane_vel_015, label='Plane', color="royalblue", linewidth=2)
    axs[0, 0].plot(x_axis, radial_vel_015, label='Radial', color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    axs[0, 0].set_ylabel('Score', fontsize=10, labelpad=1)
    axs[0, 0].text(-0.75, 0.55, '0.5X Cortex\nNoise', rotation=0, va='center', ha='center', fontsize=8,
                   transform=axs[0, 0].transAxes)
    axs[0, 0].set_xticks(x_axis)
    axs[0, 0].tick_params(axis='x', labelsize=8)
    axs[0, 0].tick_params(axis='y', labelsize=8)

    # ax2
    axs[1, 0].plot(x_axis, gaussian_vel_03, color="mediumseagreen", linewidth=2)
    axs[1, 0].plot(x_axis, plane_vel_03, color="royalblue", linewidth=2)
    axs[1, 0].plot(x_axis, radial_vel_03, color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    # axs[1, 0].set_title(r'Noise: $\sigma = 0.03 \approx 0.02$ measured - Cortex Noise')
    axs[1, 0].text(-0.75, 0.55, 'Cortex\nNoise', rotation=0, va='center', ha='center', fontsize=8,
                   transform=axs[1, 0].transAxes)
    axs[1, 0].set_xticks(x_axis)
    axs[1, 0].set_xticklabels(["X1/4", "X1/2", "1", "X2", "X4"])
    axs[1, 0].tick_params(axis='x', labelsize=8)
    axs[1, 0].tick_params(axis='y', labelsize=8)

    # ax3
    axs[2, 0].plot(x_axis, gaussian_vel_06, color="mediumseagreen", linewidth=2)
    axs[2, 0].plot(x_axis, plane_vel_06, color="royalblue", linewidth=2)
    axs[2, 0].plot(x_axis, radial_vel_06, color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    # axs[2, 0].set_title(r'Noise: $\sigma = 0.06 \approx 0.04$ measured- 2X Cortex Noise')
    axs[2, 0].text(-0.75, 0.55, '2X Cortex\nNoise', rotation=0, va='center', ha='center', fontsize=8,
                   transform=axs[2, 0].transAxes)
    axs[2, 0].set_xticks(x_axis)
    axs[2, 0].set_xticklabels(["17.5", "35", "70", "140", "280"])
    axs[2, 0].set_xlabel('Velocity [µm/frame]')
    axs[2, 0].tick_params(axis='x', labelsize=8)
    axs[2, 0].tick_params(axis='y', labelsize=8)

    ###### Intesities ####
    two_gauss_int_015noise = [0, 0.053, 0.100, 0.086, 0.072]
    plane_int_015noise = [0, 1.000, 1.000, 1.000, 1.000]
    radial_int_015noise = [0, 0.991, 0.989, 0.989, 0.991]

    two_gauss_int_03noise = [0, 0.001, 0.009, 0.013, 0.021]
    plane_int_03noise = [0, 0.999, 0.999, 0.998, 0.999]
    radial_int_03noise = [0, 0.991, 0.987, 0.987, 0.988]

    two_gauss_int_06noise = [0, 0.003, 0.001, 0.001, 0.001]
    plane_int_06noise = [0, 0.765, 0.942, 0.976, 0.982]
    radial_int_06noise = [0, 0.763, 0.930, 0.967, 0.972]

    x_axis = np.array([1, 2, 3, 4, 5])

    axs[0, 1].plot(x_axis, two_gauss_int_015noise, label='2 Gaussians', color="mediumseagreen", linewidth=2)
    axs[0, 1].plot(x_axis, plane_int_015noise, label='Plane Wave', color="royalblue", linewidth=2)
    axs[0, 1].plot(x_axis, radial_int_015noise, color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    # axs[0, 1].set_ylabel('Waviness Score')
    # axs[0, 1].set_xlabel('Velocity [pixel/frame]')
    # axs[0, 1].set_title('Score VS Velocity - No noise')
    axs[0, 1].set_xticks(x_axis)
    axs[0, 1].set_xticklabels(["0.2", "0.4", "0.6", "0.8", "1"])
    axs[0, 1].tick_params(axis='x', labelsize=8)
    axs[0, 1].tick_params(axis='y', labelsize=8)

    # ax5
    axs[1, 1].plot(x_axis, two_gauss_int_03noise, color="mediumseagreen", linewidth=2)
    axs[1, 1].plot(x_axis, plane_int_03noise, color="royalblue", linewidth=2)
    axs[1, 1].plot(x_axis, radial_int_03noise, color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    axs[1, 1].tick_params(axis='x', labelsize=8)
    axs[1, 1].tick_params(axis='y', labelsize=8)

    # ax6
    axs[2, 1].plot(x_axis, two_gauss_int_06noise, color="mediumseagreen", linewidth=2)
    axs[2, 1].plot(x_axis, plane_int_06noise, color="royalblue", linewidth=2)
    axs[2, 1].plot(x_axis, radial_int_06noise, color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    axs[2, 1].tick_params(axis='x', labelsize=8)
    axs[2, 1].tick_params(axis='y', labelsize=8)
    axs[2, 1].set_xlabel('Intesity')

    ### WIDTHS ###

    gaussians_width_015 = [0.3898, 0.1841, 0.121, 0.0964, 0.0841, 0.0622, 0.0756, 0.0798, 0.0647, 0.0821]
    plane_width_015 = [0.9854, 0.9941, 0.9987, 0.9945, 0.8798, 0.7849, 0.8155, 0.8843, 0.9228,0.9483]
    radial_width_015 = [0.9876, 0.9851, 0.9917, 0.9848, 0.9549, 0.9191, 0.9222, 0.9069, 0.889, 0.8745]

    gaussians_width_03 = [0.1017, 0.0084, 0.0093, 0.0031, 0.0087, 0.0139, 0.0214, 0.0151, 0.0257, 0.0381]
    plane_width_03 =[0.9645, 0.9936, 0.9961, 0.9732, 0.9697, 0.9812, 0.9854, 0.9865, 0.9881, 0.9873]
    radial_width_03 = [0.9751, 0.9844, 0.9889, 0.9742, 0.9467, 0.9381, 0.9379, 0.9345, 0.9321, 0.9379]

    gaussians_width_06 = [0.0, 0.0, 0.0035, 0.0, 0.0, 0.0009, 0.0016, 0.001, 0.0014, 0.0009]
    plane_width_06 = [0.8696, 0.9825, 0.9825, 0.9737, 0.9498, 0.8673, 0.7043, 0.5023, 0.2843, 0.1566]
    radial_width_06 = [0.9267, 0.9825, 0.979, 0.9626, 0.922, 0.851, 0.6849, 0.4905, 0.2756, 0.1662]

    x_axis = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    axs[0, 2].plot(x_axis, gaussians_width_015, label='2 Gaussians', color="mediumseagreen", linewidth=2)
    axs[0, 2].plot(x_axis, plane_width_015, label='Plane', color="royalblue", linewidth=2)
    axs[0, 2].plot(x_axis, radial_width_015, label='Radial', color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    axs[0, 2].set_xticks(x_axis)
    axs[0, 2].set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    axs[0, 2].tick_params(axis='x', labelsize=8)
    axs[0, 2].tick_params(axis='y', labelsize=8)

    axs[1, 2].plot(x_axis, gaussians_width_03, label='2 Gaussians', color="mediumseagreen", linewidth=2)
    axs[1, 2].plot(x_axis, plane_width_03, color="royalblue", linewidth=2)
    axs[1, 2].plot(x_axis, radial_width_03, color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    axs[1, 2].set_xticks(x_axis)
    axs[1, 2].set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    axs[1, 2].tick_params(axis='x', labelsize=8)
    axs[1, 2].tick_params(axis='y', labelsize=8)

    axs[2, 2].plot(x_axis, gaussians_width_06, label='2 Gaussians', color="mediumseagreen", linewidth=2)
    axs[2, 2].plot(x_axis, plane_width_06, color="royalblue", linewidth=2)
    axs[2, 2].plot(x_axis, radial_width_06, color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    axs[2, 2].set_xticks(x_axis)
    axs[2, 2].set_xticks([1, 4, 7, 10])
    axs[2, 2].set_xticklabels([0.3,1.3,2.3,3.3])
    #axs[2, 2].tick_params(axis='x', labelrotation=-45)
    axs[2, 2].set_xlabel('FWHM [mm]')
    axs[2, 2].tick_params(axis='x', labelsize=8)
    axs[2, 2].tick_params(axis='y', labelsize=8)

    ###### GAPS ####
    two_gaussians_gaps_015 = [0.080, 0.066, 0.065, 0.082, 0.077, 0.060, 0.070, 0.073, 0.078]
    plane_gaps_015 = [1.000, 0.967, 0.967, 0.966, 0.966, 0.965, 0.964, 0.964, 0.963]
    radial_gaps_015 = [0.989, 0.992, 0.994, 0.996, 0.995, 0.995, 0.995, 0.994, 0.995]

    two_gaussians_gaps_03noise = [0.018, 0.019, 0.015, 0.024, 0.024, 0.023, 0.026, 0.017, 0.020]
    plane_gaps_03noise = [0.999, 0.967, 0.967, 0.965, 0.964, 0.964, 0.964, 0.963, 0.963]
    radial_gaps_03noise = [0.988, 0.992, 0.993, 0.994, 0.997, 0.995, 0.995, 0.996, 0.994]

    two_gaussians_gaps_06noise = [0.002, 0.000, 0.001, 0.001, 0.001, 0.002, 0.001, 0.001, 0.001]
    plane_gaps_06noise = [0.985, 0.951, 0.948, 0.947, 0.949, 0.943, 0.947, 0.944, 0.942]
    radial_gaps_06noise = [0.971, 0.974, 0.976, 0.974, 0.977, 0.974, 0.979, 0.976, 0.975]

    x_axis = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    axs[0, 3].plot(x_axis, two_gaussians_gaps_015, label='2 Gaussians', color="mediumseagreen", linewidth=2)
    axs[0, 3].plot(x_axis, plane_gaps_015, label='Plane Wave', color="royalblue", linewidth=2)
    axs[0, 3].plot(x_axis, radial_gaps_015, color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    # axs[0, 1].set_ylabel('Waviness Score')
    # axs[0, 1].set_xlabel('Velocity [pixel/frame]')
    # axs[0, 1].set_title('Score VS Velocity - No noise')
    axs[0, 3].set_xticks(x_axis)
    axs[0, 3].set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9])
    axs[0, 3].tick_params(axis='x', labelsize=8)
    axs[0, 3].tick_params(axis='y', labelsize=8)

    # ax5
    axs[1, 3].plot(x_axis, two_gaussians_gaps_03noise, color="mediumseagreen", linewidth=2)
    axs[1, 3].plot(x_axis, plane_gaps_03noise, color="royalblue", linewidth=2)
    axs[1, 3].plot(x_axis, radial_gaps_03noise, color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    axs[1, 3].tick_params(axis='x', labelsize=8)
    axs[1, 3].tick_params(axis='y', labelsize=8)

    # ax6
    axs[2, 3].plot(x_axis, two_gaussians_gaps_06noise, color="mediumseagreen", linewidth=2)
    axs[2, 3].plot(x_axis, plane_gaps_06noise, color="royalblue", linewidth=2)
    axs[2, 3].plot(x_axis, radial_gaps_06noise, color="tomato", linewidth=2, linestyle=(0, (3, 3)))
    axs[2, 3].set_xticks([1, 4, 7, 10])  # for example
    axs[2, 3].set_xticklabels([0.07, 0.3, 0.5, 0.7])
    #axs[2, 3].tick_params(axis='x', labelrotation=-45)
    axs[2, 3].set_xlabel('Gap [mm]')
    axs[2, 3].tick_params(axis='x', labelsize=8)
    axs[2, 3].tick_params(axis='y', labelsize=8)

    for ax_row in axs:
        for ax in ax_row:
            ax.set_ylim(0, 1)
            ax.set_yticks([-0.01, 0.5, 1])
            ax.set_yticklabels(['0', '0.5', '1'])
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            ax.set_aspect((x_max - x_min) / (y_max - y_min), adjustable='box')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',  # or 'upper center', depending on your layout
        bbox_to_anchor=(0.5, 0.85),  # center horizontally, slightly below the figure
        frameon=False,
        fontsize=8,
        ncol=3,  # <-- this makes it a single row
        handlelength=2.3
    )
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].set_ylim(0, 1.01)

    # plt.tight_layout()
    plt.savefig('Basic patterns - Different Conditions .pdf', bbox_inches='tight')
    plt.show()

diff_conditions_comparison()

two_gaussians_noises = [np.float64(0.0), np.float64(0.0908), np.float64(0.0566), np.float64(0.0217), np.float64(0.0026),
                        np.float64(0.0), np.float64(0.0007), np.float64(0.0014), np.float64(0.0021), np.float64(0.0079),
                        np.float64(0.0069), np.float64(0.0157), np.float64(0.0209), np.float64(0.0302),
                        np.float64(0.0367), np.float64(0.0446), np.float64(0.0481), np.float64(0.051),
                        np.float64(0.055), np.float64(0.0553), np.float64(0.0591)]
plane_noises = [1, 1, 1, 0.999, 0.999, 0.994, 0.982, 0.958, 0.915, 0.843, 0.747, np.float64(0.621), np.float64(0.5107),
                np.float64(0.4194), np.float64(0.3297), np.float64(0.2648), np.float64(0.226), np.float64(0.1887),
                np.float64(0.1765), np.float64(0.177), np.float64(0.158)]
radial_noises = [0.986, 0.989, 0.991, 0.989, 0.983, 0.983, 0.977, 0.953, 0.904, 0.836, np.float64(0.7293),
                 np.float64(0.6332), np.float64(0.5088), np.float64(0.3882), np.float64(0.3212), np.float64(0.2552),
                 np.float64(0.2085), np.float64(0.1575), np.float64(0.1529), np.float64(0.1367), np.float64(0.1219)]

