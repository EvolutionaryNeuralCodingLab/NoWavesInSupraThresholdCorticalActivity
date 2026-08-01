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

theta = 15  ## Theta threshold for uniform continuity

plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42


class MatlabToDff:
    def __init__(self, data):
        self.dff = data['dFF']
        self.N = data['dFF'].shape[0]
        self.M = data['dFF'].shape[1]
        self.frames = data['dFF'].shape[2]

    ## enhance Widefield CA imaging output . Specific editing of our data.
    def enhance(self):
        # Replace -1 values with 0 in each frame
        self.dff = np.where(self.dff == -1, 0, self.dff)
        self.dff[self.dff < -0.9] = 0


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

        brain_mask = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/brain_mask.npy')
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
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.iter = iterations
        self.lim = lim

    def plot_frames(self):

        #frame_indices =  [10,20,30,40,50,60]
        #frame_indices = [i for i in range(0,110,11)] # For example 2 in retina supp
        #frame_indices =[15,36,99,148,175,203,245,278,354]   # Tim murphy cortex

        #frame_indices = [0,7,16,21,28,35 ,40,48,54,68]
        frame_indices = [6*i for i in range(20)]

        first_half = frame_indices[:len(frame_indices) // 2]
        second_half = frame_indices[len(frame_indices) // 2:]

        # Load outer line
        outer_line_rgb = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/outer_line_rgb.npy')
        outer_line_rgba = np.ones((*outer_line_rgb.shape[:2], 4))
        line_mask = np.all(outer_line_rgb < 1.0, axis=2)
        outer_line_rgba[..., :3][line_mask] = 0.392
        outer_line_rgba[..., 3] = 0.0
        outer_line_rgba[..., 3][line_mask] = 1.0

        brain_mask = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/brain_mask.npy')
        brain_mask_bool = brain_mask.astype(bool)
        eroded_mask = binary_erosion(brain_mask_bool, iterations=1)
        brain_mask = eroded_mask.astype(np.uint8)[:,64:]

        figsize_cm = (5, 5)
        figsize_in = tuple(x / 2.54 for x in figsize_cm)
        fig = plt.figure(figsize=figsize_in, dpi=600)

        # Add 1 row for gap → total 7 rows
        cols = len(frame_indices) // 2
        gs = gridspec.GridSpec(
            2, cols + 1,
            width_ratios=[1] * cols + [0.03],
            height_ratios=[0.1, 0.1],  # row 3 is the gap
            hspace=0.35, wspace=0
        )

        # Top half (rows 0–2)
        top1_axes = [fig.add_subplot(gs[0, i]) for i in range(cols)]

        # Bottom half (rows 4–6)
        top2_axes = [fig.add_subplot(gs[1, i]) for i in range(cols)]
        from scipy.ndimage import gaussian_filter

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
        def plot_set(frame_indices, top_axes):
            for i, frame_idx in enumerate(frame_indices):
                snapshot = self.dff[:, :, frame_idx]
                snapshot = np.where(brain_mask == 1, snapshot, np.nan)
                cmap.set_bad(color="white")

                ax_top = top_axes[i]
                im = ax_top.imshow(snapshot, cmap=self.color_map, vmin=0, vmax=1)
                #im = ax_top.imshow(snapshot, cmap= cmap, vmin=-np.pi, vmax=np.pi)

                #ax_top.imshow(outer_line_rgba)
                ax_top.set_aspect('equal')
                ax_top.axis('off')



        plot_set(first_half, top1_axes)
        plot_set(second_half, top2_axes)

        plt.savefig('Ca OF and Momentum example .pdf', bbox_inches='tight', dpi = 600)

        plt.show()

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
        brain_mask = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/brain_mask.npy')[:,:64]
        # brain_mask = brain_mask[:, :64]
        outer_line_rgb = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/outer_line_rgb.npy')
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

        outer_line_rgb = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/outer_line_rgb.npy')

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
            brain_mask = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/brain_mask.npy')
            outer_line_rgb = np.load(
                '/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/outer_line_rgb.npy')

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
        plt.savefig(f'{self.N}x{self.M}x{self.frames} {self.title}.pdf', dpi=400)
        # plt.savefig('Retina2.pdf')
        plt.show()


def data_type(type):
    if type == '1 gaussian':
        dff1, params = create_gaussians(N=64, M=128, frames=60, num_gaus=1, x0=32, y0=70, sd0=16, t0_0=30, sdT0=10)
        title = fr'1 Gaussian $\sigma_x$={params[6]} , $\sigma_T$={params[8]}'

        return dff1, title

    if type == '1 gaussian moving':
        dff1, params = create_gaussians_moving(N=64, M=128, frames=60, num_gaus=1, x0=32, y0=100, sd0=14, t0_0=30,
                                               sdT0=14)
        title = fr'1 Moving Gaussian $\sigma$={params[6]} '  # $\sigma_x$={params[6]} , $\sigma_T$={params[8]}

        return dff1, title

    if type == '2 gaussian 2sig':
        # dff1 , params = create_gaussians(N=64, M=64, frames=50, num_gaus=2, x0=20+17.5, y0=32, sd0=7, t0_0=19, sdT0=7 \
        #                                                        , x1=20, y1=32,sd1=7, t0_1=19+17.5 ,sdT1=7)

        dff1, params = create_gaussians(N=64, M=128, frames=50, num_gaus=2, x0=20 + 14, y0=100, sd0=7, t0_0=19, sdT0=7,
                                        x1=20, y1=100, sd1=7, t0_1=19 + 14, sdT1=7)

        dff2, params2 = create_gaussians(N=64, M=128, frames=50, num_gaus=2, x0=20 + 14, y0=50, sd0=7, t0_0=19, sdT0=7,
                                         x1=20, y1=50, sd1=7, t0_1=19 + 1004, sdT1=7)

        dff2[:, :, 20:33] = create_patterns(N=64, M=128, frames=13, pattern='cont', x0=30, y0=50, sd0=7, u=1, v=0,
                                            rad_spd=1, rad_width=5)

        decrease, params2 = create_gaussians(N=64, M=128, frames=50, num_gaus=1, x0=42, y0=50, sd0=7, t0_0=19 + 14,
                                             sdT0=7, x1=20, y1=50, sd1=7, t0_1=19 + 14, sdT1=7)
        print(decrease.shape)

        dff2[:, :, 33:] = decrease[:, :, 33:]
        # dff2[:,:,19:33] = create_patterns(N=64, M=128, frames=14, pattern='cont', x0=1, y0=34, sd0=7, u=-11, v=0, rad_spd=1, rad_width=5)

        # dff1 +=create_patterns(N=64, M=128, frames=50, pattern='cont', x0=20, y0=50, sd0=7, u=0.8, v=0, rad_spd=1, rad_width=5)

        dff1 += dff2

        # dff1[:,30,:]=1
        # dff1[:,44,:]=1

        title = r'2 Gaussians $\Delta$X=$\Delta$T=2.5$\sigma$, $\sigma$=7'

        return dff1, title

    if type == '2 gaussian and plane':
        # dff1 , params = create_gaussians(N=64, M=64, frames=50, num_gaus=2, x0=20+17.5, y0=32, sd0=7, t0_0=19, sdT0=7 \
        #                                                        , x1=20, y1=32,sd1=7, t0_1=19+17.5 ,sdT1=7)

        sigma = 8

        if sigma == 6:
            dff1, params = create_gaussians(N=64, M=128, frames=55, num_gaus=2, x0=20 + 15, y0=100, sd0=6, t0_0=19,
                                            sdT0=6, x1=20, y1=100, sd1=6, t0_1=19 + 15, sdT1=6)

            dff2, params2 = create_gaussians(N=64, M=128, frames=55, num_gaus=2, x0=20 + 15, y0=50, sd0=6, t0_0=19,
                                             sdT0=6, x1=20, y1=50, sd1=6, t0_1=19 + 1004, sdT1=6)
            dff2[:, :, 19:31] = create_patterns(N=64, M=128, frames=12, pattern='cont', x0=32, y0=50, sd0=6, u=1, v=0,
                                                rad_spd=1, rad_width=5)

            decrease, params2 = create_gaussians(N=64, M=128, frames=55, num_gaus=1, x0=44, y0=50, sd0=6, t0_0=19 + 15,
                                                 sdT0=6, x1=20, y1=50, sd1=6, t0_1=19 + 1000, sdT1=6)
            dff2[:, :, 31:] = decrease[:, :, 31:]

        if sigma == 7:
            dff1, params = create_gaussians(N=64, M=128, frames=55, num_gaus=2, x0=20 + 17, y0=100, sd0=7, t0_0=19,
                                            sdT0=7, x1=20, y1=100, sd1=7, t0_1=19 + 17, sdT1=7)
            dff2, params2 = create_gaussians(N=64, M=128, frames=55, num_gaus=2, x0=20 + 17, y0=50, sd0=7, t0_0=19,
                                             sdT0=7, x1=20, y1=50, sd1=7, t0_1=19 + 1004, sdT1=7)
            dff2[:, :, 20:37] = create_patterns(N=64, M=128, frames=17, pattern='cont', x0=26, y0=50, sd0=7, u=1, v=0,
                                                rad_spd=1, rad_width=5)
            decrease, params2 = create_gaussians(N=64, M=128, frames=55, num_gaus=1, x0=42, y0=50, sd0=7, t0_0=19 + 17,
                                                 sdT0=7, x1=20, y1=50, sd1=7, t0_1=19 + 1000, sdT1=7)
            dff2[:, :, 37:] = decrease[:, :, 37:]

        if sigma == 8:
            dff1, params = create_gaussians(N=64, M=128, frames=70, num_gaus=2, x0=20 + 20, y0=100, sd0=8, t0_0=20,
                                            sdT0=8, x1=20, y1=100, sd1=8, t0_1=20 + 20, sdT1=8)

            dff2, params2 = create_gaussians(N=64, M=128, frames=70, num_gaus=2, x0=20 + 20, y0=50, sd0=8, t0_0=20,
                                             sdT0=8, x1=20, y1=50, sd1=8, t0_1=19 + 1004, sdT1=8)

            dff2[:, :, 20:40] = create_patterns(N=64, M=128, frames=20, pattern='cont', x0=24, y0=50, sd0=8, u=1, v=0,
                                                rad_spd=1, rad_width=5)

            decrease, params2 = create_gaussians(N=64, M=128, frames=70, num_gaus=1, x0=44, y0=50, sd0=8, t0_0=20 + 20,
                                                 sdT0=8, x1=20, y1=50, sd1=8, t0_1=19 + 1000, sdT1=8)
            dff2[:, :, 40:] = decrease[:, :, 40:]

        # dff2[:,:,19:33] = create_patterns(N=64, M=128, frames=14, pattern='cont', x0=1, y0=34, sd0=7, u=-11, v=0, rad_spd=1, rad_width=5)

        # dff1 +=create_patterns(N=64, M=128, frames=50, pattern='cont', x0=20, y0=50, sd0=7, u=0.8, v=0, rad_spd=1, rad_width=5)

        dff1 += dff2

        # dff1[:,30,:]=1
        # dff1[:,44,:]=1

        title = r'2 Gaussians $\Delta$X=$\Delta$T=2.5$\sigma$, $\sigma$=7'

        return dff1, title

    if type == '2 gaussian 2.25sig':
        dff1, params = create_gaussians(N=64, M=128, frames=65, num_gaus=2, x0=32, y0=70 - 18, sd0=16, t0_0=22 + 22.5,
                                        sdT0=10, x1=32, y1=70 + 18, sd1=16, t0_1=22, sdT1=10)

        # dff1 , params = create_gaussians(N=64, M=128, frames=50, num_gaus=2, x0=30, y0=90, sd0=17, t0_0=16, sdT0=7 \
        #                                                       , x1=30, y1=90-32,sd1=17, t0_1=16+12 ,sdT1=7)

        # Parameters for soft easing
        # threshold = 0.25
        # steepness = 20  # Higher = sharper transition, like a soft cutoff

        # mask = 1 / (1 + np.exp(-steepness * (dff1 - threshold)))        # Sigmoid-based soft mask
        # dff1 = dff1 * mask

        title = r'2 Gaussians $\Delta$X=$\Delta$T=2.25$\sigma$, $\sigma$=7'

        return dff1, title

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


    if type == '2 gaussian example':
        dff1, params = create_gaussians(N=64, M=128, frames=65, num_gaus=2, x0=30 - 12.5, y0=64, sd0=10, t0_0=20,
                                        sdT0=10 \
                                        , x1=30 + 12.5, y1=64, sd1=10, t0_1=20 + 25, sdT1=10)

        title = r'2 Gaussians $\Delta$X=$\Delta$T=2.5$\sigma$, $\sigma$=6'

        return dff1, title

    if type == '2 gaussian moving':
        dff1, params = create_gaussians_moving(N=64, M=128, frames=60, num_gaus=2, x0=5, y0=70, sd0=10, t0_0=19, sdT0=10 \
                                               , x1=5, y1=60 + 25, sd1=10, t0_1=19 + 25, sdT1=10)

        title = r'2 Moving Gaussians $\Delta$X=$\Delta$T=2.5$\sigma$, $\sigma$=6'
        return dff1, title

    if type == 'plane':
        dff1 = create_patterns(N=128, M=64, frames=40, pattern='plane', x0=32, y0=90, sd0=8, u=0, v=1, rad_spd=1,rad_width=5)

        #dff1 = create_patterns(N=64, M=64, frames=80, pattern='plane', x0=-5, y0=0, sd0=8, u=0, v=0.5 ,rad_spd=1, rad_width=4)
        # dff1 = create_patterns(N=128, M=128, frames=110, pattern='plane', x0=-20, y0=0, sd0=8, u=0, v=1 ,rad_spd=1, rad_width=4)

        title = 'Plane Wave'

        return dff1, title

    if type == 'radial':
        dff1 = create_patterns(N=128, M=64, frames=49, pattern='radial', x0=32, y0=80, sd0=2, u=0, v=1, rad_spd=0.5,rad_width=3)

        # dff1 = create_patterns(N=64, M=64, frames=23, pattern='radial', x0=32, y0=32, sd0=2, u=0, v=1, rad_spd=1,rad_width=5)

        #dff1 = create_patterns(N=128, M=64, frames=50, pattern='radial', x0=30, y0=138, sd0=2, u=0, v=1,rad_spd=3, rad_width=5)

        title = 'Radial Wave'

        return dff1, title

    if type == 'two radials':
        dff1 = create_patterns(N=128, M=64, frames=40, pattern='radial', x0=20, y0=110, sd0=1, u=0, v=0.5, rad_spd=0.5,
                               rad_width=2)
        dff1 += create_patterns(N=128, M=64, frames=40, pattern='radial', x0=45, y0=20, sd0=1, u=0, v=0.5, rad_spd=0.5,
                                rad_width=2)
        dff1 += create_patterns(N=128, M=64, frames=40, pattern='radial', x0=35, y0=60, sd0=1, u=0, v=0.5, rad_spd=0.5,
                                rad_width=2)

        title = 'Radial Wave'

        return dff1, title

    if type == 'radial with gaps':
        dff1 = create_patterns(N=128, M=64, frames=49, pattern='radial', x0=32, y0=80, sd0=2, u=0, v=0.5, rad_spd=0.5,
                               rad_width=3)

        # dff1[16,11:20,:] = 0
        # dff1[15,18:26,:] = 0
        # dff1[16,24:31,:] = 0
        # dff1[17, 30:35,:] = 0
        # dff1[16, 33:40,:] = 0

        dff1[67, 0:63, :] = 0
        dff1[68, 0:63, :] = 0

        # dff1[94,10:60,:]=0
        dff1[95, 0:63, :] = 0

        title = 'Radial Wave with gaps'

        return dff1, title

    if type == 'spiral':
        video_path = "/Users/arielrom/Desktop/תואר שני/Thesis/AnalyzedData/spiral.mp4"
        mp4_data = MP4ToDff(video_path)
        mp4_data.mp4_video_to_numpy_gray()
        #print(mp4_data.dff)

        # mp4_data.dff = mp4_data.dff[:,:,1200:1380] #1100,1400   ,  530,710
        # mp4_data.dff = mp4_data.dff[:,:,1220:1400] #1100,1400   ,  530,710
        mp4_data.dff = mp4_data.dff[145:, 65:-80, 1330:1530]  # 1100,1400   ,  530,710 , 1240:1380
        mp4_data.dff = decrease_frame_rate(mp4_data.dff, 10, 5)
        mp4_data.frames = mp4_data.dff.shape[2]

        mp4_data.dff = mp4_data.dff - 60
        mp4_data.dff[mp4_data.dff > 200] = 0

        mp4_data.dff = normalize_data(mp4_data.dff)

        title = 'Spiral Wave'
        return mp4_data.dff, title

    if type == '2 diff gaussians moving':
        dff2, params = create_gaussians_moving(N=70, M=70, frames=50, num_gaus=1, x0=20, y0=50, sd0=5, t0_0=10, sdT0=5)

        dff1, params = create_gaussians_moving(N=70, M=70, frames=50, num_gaus=2, x0=5, y0=25, sd0=5, t0_0=10, sdT0=5,
                                               x1=25, y1=5, sd1=5, t0_1=10, sdT1=5,
                                               x2=None, y2=None, sd2=None, t0_2=None, sdT2=None)

        dff = dff1 + dff2

        title = fr'2 Gaussians $\sigma_x$={params[6]} , $\sigma_T$={params[8]}'

        return dff, title

    if type == 'cont':
        dff1 = create_patterns(N=50, M=50, frames=70, pattern='cont', x0=20, y0=-10, sd0=7, u=0, v=1, rad_spd=0.3,
                               rad_width=3)
        title = 'cont'

        return dff1, title
        # dff1= create_pattenrs(N=60, M=60, frames=70, pattern='cont', x0=35, y0=-20, sd0=8, u=0, v=1,rad_spd=0.3, rad_width=4)

    if type == 'radial with gaps':
        dff1 = create_patterns(N=128, M=128, frames=23, pattern='radial', x0=32, y0=80, sd0=1, u=0, v=1, rad_spd=1,
                               rad_width=2)

        # dff1[16,11:20,:] = 0
        # dff1[15,18:26,:] = 0
        # dff1[16,24:31,:] = 0
        # dff1[17, 30:35,:] = 0
        # dff1[16, 33:40,:] = 0

        dff1[65, 10:60, :] = 0
        dff1[66, 10:60, :] = 0

        # dff1[94,10:60,:]=0
        dff1[95, 10:60, :] = 0

        title = 'Radial Wave'

        return dff1, title

    if type == 'cortex':
        x = scipy.io.loadmat(
            "/Users/arielrom/Desktop/תואר שני/Thesis/AnalyzedData/Early Tryings/spont_4000f_MMStack_Pos0_1.mat")
        mat_data = MatlabToDff(x)
        mat_data.enhance()
        dff1 = normalize_data(mat_data.dff[:, :, 50:1950])
        # dff1 = normalize_data(mat_data.dff[:, :, 450:510])

        # dff1 = mat_data.dff[120:270,10:160,245:305] ##(426,511) ##(0,450) , (200,400)

        title = 'spont_4000f_MMStack_Pos0_1'

        return dff1, title
        # return decrease_frame_rate(dff1,10,5), title

        # return dff1[:,:,175:245] , title
        # return dff1[:,:,285:305] , title

    if type == 'SD':
        video_path = "/Users/arielrom/Desktop/תואר שני/Thesis/AnalyzedData/FullCode/SD_WAVE.mp4"
        mp4_data = MP4ToDff(video_path)

        mp4_data.mp4_video_to_numpy_gray()
        # np.save("SD NUMPY",mp4_data.dff)
        # mp4_data.dff = mp4_data.dff[200:600,300:700,350:560]

        # mp4_data.dff = normalize_data(mp4_data.dff)

        dff1 = mp4_data.dff
        # np.save("SD_Normalized",dff1)
        # dff1 = np.load('SD_Normalized.npy')
        dff1 = dff1[420:700, 400:680, 250:-300]
        # savemat('my_array.mat', {'dff': decrease_frame_rate(dff1, 30, 10)})

        print(decrease_frame_rate(dff1, 20, 10).shape)

        title = 'Spreading Depression '

        # return dff1 , title
        return decrease_frame_rate(dff1, 20, 10), title

    if type == 'retina tif':
        def load_tif_to_numpy(tif_file_path):
            # Load the TIFF file as a NumPy array
            with tifffile.TiffFile(tif_file_path) as tif:
                # If the TIFF file has multiple pages (e.g., different channels or timepoints), use 'asarray'
                data = tif.asarray()

            # Example: If it's a 3D file, the shape will be (frames, height, width)
            # For multi-channel, it may be 4D (channels, frames, height, width) or (frames, height, width, channels)

            return data

        tif_file_path = '/Users/arielrom/Desktop/תואר שני/Thesis/AnalyzedData/TifConvert/211119_n3_r4_mov_10_frames 988-1517 WT (Mov 1).tif'
        dff1 = load_tif_to_numpy(tif_file_path)
        dff1 = np.transpose(dff1, (1, 2, 0))

        # dff1 = np.transpose(dff1, (2, 3, 0,1))
        # dff1 = np.dot(dff1[..., :3], [0.2989, 0.5870, 0.1140])

        print(dff1.shape)

        title = 'other'
        return dff1, title




brain_mask = np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/brain_mask.npy')


brain_mask = brain_mask#[:, :64]

def top_results():
    datasets = [
        ("54MRL_16to19", np.load(
            '/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/54MRL/54MRL_wf_raw_data_16to19.npy')[:,
                         :, 3238:3294]),
        ("63MR_0to3",
         np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/63MR/63MR_wf_raw_data_0to3.npy')[:,
         :, 2934:2972]),
        ("63MR_12to15",
         np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/63MR/63MR_wf_raw_data_12to15.npy')[
         :, :, 1806:1856]),
        ("63MR_24to27",
         np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/63MR/63MR_wf_raw_data_24to27.npy')[
         :, :, 547:601]),
        ("203MN_20to23", np.load(
            '/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/203MN/203MN_wf_raw_data_20to23.npy')[:,
                         :, 1011:1035]),
        ("206FRL_24to27", np.load(
            '/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/206FRL/206FRL_wf_raw_data_24to27.npy')[:,
                          :, 2388:2453]),
        ("218MN_16to19", np.load(
            '/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/218MN/218MN_wf_raw_data_16to19.npy')[:,
                         :, 462:522]),
        ("21ML_0to3",
         np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/21ML/21ML_wf_raw_data_0to3.npy')[:,
         :, 1745:1770]),
        ("21ML_4to7_A",
         np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/21ML/21ML_wf_raw_data_4to7.npy')[:,
         :, 1931:2007]),
        ("21ML_4to7_B",
         np.load('/Users/arielrom/Desktop/תואר שני/Thesis/Waves Detection Algorithm/21ML/21ML_wf_raw_data_4to7.npy')[:,
         :, 2452:2595])
    ]

    space = 6
    scale = 0.05

    # ----------------------------
    # 3. PROCESS EACH DATASET
    # ----------------------------

    for name, dff in datasets:
        print(f"\n==============================")
        print(f"Processing dataset: {name}")
        print(f"==============================\n")

        # ---- Preprocess ----
        data = PreDataProcessing(dff)
        data.resize(128, 128)

        # ---- Optical flow ----
        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase = False)

        # ---- Compute waviness ----
        data.calculate_waveness(type='retina')

        # ---- Plot & Save ----
        display = Display(data)
        display.title = name  # To save PDF with dataset name
        display.full_analysis_3columns(
            space=space,
            scale=scale,
            data_type="cortex"
        )



top_results()