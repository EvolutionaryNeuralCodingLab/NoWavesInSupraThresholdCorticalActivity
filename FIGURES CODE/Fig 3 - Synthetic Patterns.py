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
import os


#### Algorithm parameters###
alpha = 0.3   ## Optic Flow Horn-Schunck alpha parameter
iterations = 150  ## Number of iterations
n = 3  ## neighborhood size (2n+1)x(2n+1) around each pixel
lim = 0.5  ## Threshold of Wavness values for final score
beta = 0.035

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

            # Call the find_max_min function on the smoothed gradients
            score = find_max_min(profile_1d, sigma=sigma, prominence=prominence, lim_up=lim_up, lim_down=lim_down)

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

                    self.waveness[h, j, 3] = ratios[h, j] * wave_front_map[h, j]  # * spatial_coherence

                    self.mask[h, j] = 1


                else:
                    self.waveness[h, j, 3] = 0


        self.Waveness = True


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
        dff1 = np.load("spiral example.npy")
        title = 'Spiral Wave'

        return dff1, title




def bandpass_filter_video(video, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')

    # Apply filter along the last axis (time)
    filtered = filtfilt(b, a, video, axis=-1)

    return filtered


def cyclic_hsv_cmap(n=256, rotation=0.25):
    hue = (np.linspace(0, 1, n + 1) + rotation) % 1

    saturation = 0.7 + 0.3 * np.sin(2 * np.pi * hue)
    value = 0.85 + 0.15 * np.cos(2 * np.pi * hue)

    hsv = np.stack((hue, saturation, value), axis=1)
    rgb = hsv_to_rgb(hsv.reshape(1, -1, 3)).squeeze()[:-1]

    return LinearSegmentedColormap.from_list("cyclic_hsv", rgb)


def phase_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
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
            "#d45568",
        ],
        N=1024,
    )


def hide_axis(ax):
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)


def ratio_over_lim(values, mask, threshold):
    n_active = np.count_nonzero(mask)
    return np.count_nonzero(values > threshold) / n_active if n_active else 0


def apply_brain_mask(data, brain_mask):
    mask = brain_mask

    while mask.ndim < data.dff.ndim:
        mask = mask[..., None]

    data.dff *= mask


def analyze_dataset(dff, brain_mask, preprocess=None):
    data = PreDataProcessing(dff)

    if preprocess is not None:
        preprocess(data)

    apply_brain_mask(data, brain_mask)

    data = FlowAnalyze(data)
    data.horn_schunck_flow(
        alpha=alpha,
        num_iter=iterations,
        phase=False,
    )
    data.calculate_waveness(type="cortex")

    return data


def plot_source_image(ax, image_path):
    ax.imshow(mpimg.imread(image_path))
    ax.axis("off")


def plot_histogram(ax, values, threshold, bins, color, edge_color, line_width):
    ax.hist(
        values,
        bins=bins,
        range=(0, 1),
        color=color,
        edgecolor=edge_color,
        linewidth=line_width,
        rwidth=1,
    )

    ax.set(
        xlim=(0, 1),
        ylim=(0, len(values)),
        yticks=[0, len(values)],
        yticklabels=[0, 1],
    )

    ax.axvline(
        threshold,
        color="dimgrey",
        linewidth=1,
        linestyle="--",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(labelsize=8)

    x_range = np.diff(ax.get_xlim())[0]
    y_range = np.diff(ax.get_ylim())[0]

    if y_range > 0:
        ax.set_aspect(x_range / y_range, adjustable="box")


def plot_flow(ax, data, space, scale, color, width):
    vectors = data.sum_phase_space[:, :, -1, :].copy()

    zero_vectors = np.all(vectors == 0, axis=-1)
    vectors[zero_vectors] = np.nan

    plot_quiver(
        ax,
        vectors,
        spacing=space,
        scale=scale,
        color=color,
        width=width,
    )

    ax.set_ylim(data.dff.shape[0], 0)
    ax.set_aspect("equal", adjustable="box")
    hide_axis(ax)


def plot_waviness(ax, data, threshold, cmap, grey=100):
    rgba_background = np.zeros((*data.mask.shape, 4), dtype=float)
    rgba_background[..., :3] = grey / 255
    rgba_background[..., 3] = np.where(data.mask == 1, 0.1, 0)

    alpha_map = np.where(
        data.waveness[:, :, 3] >= threshold,
        data.waveness[:, :, 3],
        0,
    )

    ax.imshow(rgba_background)
    ax.imshow(
        data.waveness[:, :, 0],
        cmap=cmap,
        vmin=-np.pi,
        vmax=np.pi,
        alpha=alpha_map,
    )

    hide_axis(ax)


def calculate_phase(data, original_dff, config):
    if config.get("bandpass"):
        fs = config.get("fs", 35.0)
        lowcut = config.get("lowcut", 2)
        highcut = config.get("highcut", 8.0)

        phase_input = bandpass_filter_video(
            original_dff,
            lowcut,
            highcut,
            fs,
        )
    else:
        phase_input = data.dff - data.dff.mean()

    return np.angle(hilbert(phase_input, axis=-1))


def plot_phase(ax, data, original_dff, frame, cmap, config):
    phase = calculate_phase(data, original_dff, config)

    masked_phase = np.ma.masked_where(
        ~data.mask.astype(bool),
        phase[:, :, frame],
    )

    ax.imshow(
        masked_phase,
        cmap=cmap,
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax.axis("off")


def preprocess_spiral(data):
    data.resize(135, 135)
    data.dff = data.dff[7:, 25:89, :]
    data.dff[:70, :, :] = 0

    shift_row = 4
    shift_col = 3

    shifted = np.zeros_like(data.dff)
    shifted[:-shift_row, shift_col:, :] = data.dff[
        shift_row:,
        :-shift_col,
        :,
    ]

    data.dff = shifted


def plot_dataset_row( axes, config, brain_mask, space,scale, waviness_cmap, phase_map, settings):
    original_dff, _ = data_type(config["data_type"])

    data = analyze_dataset( original_dff, brain_mask, preprocess=config.get("preprocess"))

    values = data.waveness[:, :, 3][data.mask == 1]

    ratio = ratio_over_lim( values, data.mask, settings["lim"])

    print(f'{config["label"]}: {ratio:.4f}')

    plot_histogram( axes[0], values, threshold=settings["lim"], bins=settings["bins"], color=settings["hist_color"], edge_color=settings["edge_color"], line_width=settings["hist_width"])

    plot_flow(axes[1], data, space=space, scale=scale, color=settings["quiver_color"], width=settings["quiver_width"])

    plot_waviness( axes[2], data, threshold=settings["lim"], cmap=waviness_cmap, grey=settings["grey"])

    plot_phase( axes[3], data, original_dff, frame=config["phase_frame"], cmap=phase_map, config=config)


def figure_plot(space, scale):


    datasets = [

        {
            "data_type": "plane",
            "label": "plane",
            "phase_frame": 20,
        },
        {
            "data_type": "1 gaussian",
            "label": "1 gaussian",
            "phase_frame": 35,
        },
        {
            "data_type": "spiral",
            "label": "spiral",
            "phase_frame": 35,
            "preprocess": preprocess_spiral,
        },
        {
            "data_type": "2 gaussian 2.25sig",
            "label": "2 gaussians",
            "phase_frame": 35,
        },
        {
            "data_type": "radial",
            "label": "radial",
            "phase_frame": 22,
        },
        {
            "data_type": "3 gaussian",
            "label": "3 gaussian",
            # Replace this if the 3-Gaussian illustration has its own file.
            "phase_frame": 12,
            "bandpass": True,
            "fs": 35.0,
            "lowcut": 2,
            "highcut": 8.0,
        },
        {
            "data_type": "radial with gaps",
            "label": "radial with gaps",
            "phase_frame": 22,
        },
        {
            "data_type": "1 gaussian moving",
            "label": "1 gaussian moving",
            "phase_frame": 31,
        }
    ]

    settings = {
        "lim": lim,
        "bins": 10,
        "hist_color": "#d6b8a8",
        "edge_color": "#000000",
        "hist_width": 0.5,
        "quiver_color": "black",
        "quiver_width": 0.007,
        "grey": 100,
    }

    figsize_cm = (18, 18)
    figsize_inches = tuple(value / 2.54 for value in figsize_cm)

    fig, axes = plt.subplots( nrows=4, ncols=8, figsize=figsize_inches )

    waviness_cmap = cyclic_hsv_cmap()
    waviness_cmap.set_bad("white")

    phase_map = phase_cmap()

    brain_mask = np.load(f"brain_mask.npy")[:, :64]

    # Each dataset occupies four consecutive panels.
    # First four datasets are placed on the left half,
    # and the next four on the right half.
    panel_locations = [
        (0, 0),
        (0, 4),
        (1, 0),
        (1, 4),
        (2, 0),
        (2, 4),
        (3, 0),
        (3, 4),
    ]

    for config, (row, start_column) in zip(
        datasets,
        panel_locations,
    ):
        row_axes = axes[row, start_column:start_column + 4]

        plot_dataset_row(
            axes=row_axes,
            config=config,
            brain_mask=brain_mask,
            space=space,
            scale=scale,
            waviness_cmap=waviness_cmap,
            phase_map=phase_map,
            settings=settings,
        )

    plt.savefig("Basic Examples.pdf",bbox_inches="tight",dpi=1000,)
    plt.show()



figure_plot(space = 5, scale = 0.15)

