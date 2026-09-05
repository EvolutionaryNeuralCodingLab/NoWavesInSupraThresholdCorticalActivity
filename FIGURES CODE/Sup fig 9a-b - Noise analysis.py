import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.animation as animation

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, label

from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb

from Algos.Display import plot_quiver
from Algos.Create_Patterns import create_patterns
from Algos.Data_Processing import resize
from Algos.Horn_Schunck import horn_schunck, horn_schunck_phase

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

    def add_noise(self, std):
        for t in range(self.frames):
            self.dff[:, :, t] += np.random.normal(0, std, size=(self.N, self.M))


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

    def full_analysis_3columns(self, space, scale, data_type, plot = False):

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

        if plot == True:
            plt.show()




def compare_basic_patterns_noise(noise, N=128, M=64, frames=65, x0=32, y0=70 - 18, sd0=16, t0_0=22 + 22.5, sdT0=10,
                                 x1=32, y1=70 + 18, sd1=16, t0_1=22, sdT1=10, plot_data=False):
    noise_values = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                    0.17, 0.18, 0.19, 0.2]

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
    for std in noise_values:
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
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase=False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'2 gaussians , noise added = {std}'
        if plot_data == True:
            display.plot_data()

        flattened_values = data.waveness[:, :, 3].flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_2gaussian.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("gaussians:", ratios_2gaussian)

    ratios_plane = []
    for std in noise_values:
        dff = create_patterns(N=N, M=M, frames=frames, pattern='plane', x0=32, y0=90, sd0=8, u=0, v=1,
                              rad_spd=1, rad_width=5)

        data = PreDataProcessing(dff)
        if noise == True:
            data.add_noise(std)
            data.dff = np.clip(data.dff, a_min=0, a_max=1)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask
        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase=False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'plane , noise added = {std}'
        if plot_data == True:
            display.plot_data()

        flattened_values = data.waveness[:, :, 3].flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_plane.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("plane:", ratios_plane)

    ratios_radial = []
    for std in noise_values:
        dff = create_patterns(N=N, M=M, frames=frames, pattern='radial', x0=M / 2, y0=N / 2, sd0=8, u=0, v=1,rad_spd=1, rad_width=5)
        data = PreDataProcessing(dff)
        if noise == True:
            data.add_noise(std)

            data.dff = np.clip(data.dff, a_min=0, a_max=1)
        while brain_mask.ndim < data.dff.ndim:
            brain_mask = np.expand_dims(brain_mask, axis=-1)

        data.dff *= brain_mask

        data = FlowAnalyze(data)
        data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase=False)
        data.calculate_waveness(type='retina')
        display = Display(data)
        display.title = f'radial , noise added = {std}'
        if plot_data == True:
            display.plot_data()

        flattened_values = data.waveness[:, :, 3].flatten()
        ratio = ratio_over_lim(lim, flattened_values, data.mask)
        ratios_radial.append(np.round(ratio, 4))
        display.full_analysis_3columns(space=3, scale=0.2, data_type="cortex")

    print("radial:", ratios_radial)

### For 1 time calculation of scores in different noise levels:
compare_basic_patterns_noise(noise = True)




### Scores

two_gaussians_spatial = [0, 0.091, 0.057, 0.022, 0.003, 0, 0.001, 0.001, 0.002, 0.008, 0.007, 0.016, 0.021, 0.030, 0.037, 0.045, 0.048, 0.051, 0.055, 0.055, 0.059]
plane_spatial = [1, 1, 1, 0.999, 0.999, 0.994, 0.982, 0.958, 0.915, 0.843, 0.747, 0.621, 0.511, 0.419, 0.330, 0.265, 0.226, 0.189, 0.176, 0.177, 0.158]
radial_spatial = [0.986, 0.989, 0.991, 0.989, 0.983, 0.983, 0.977, 0.953, 0.904, 0.836, 0.729, 0.633, 0.509, 0.388, 0.321, 0.255, 0.209, 0.158, 0.153, 0.137, 0.122]


def gaussian_sigma(sigma):
    if sigma == 1 :
        plane_noise = [0.0057, 0.0081, 0.0108, 0.0137, 0.0167, 0.0197, 0.0227, 0.0256, 0.0287, 0.0317, 0.0347, 0.0377, 0.0406, 0.0437, 0.0465, 0.0495, 0.0524, 0.0553, 0.0583, 0.0611, 0.0641, 0.0672, 0.07, 0.0727, 0.0756, 0.0785, 0.0812, 0.0841, 0.0869, 0.0898, 0.0927, 0.0953, 0.0981, 0.1008, 0.1036, 0.1066, 0.1092, 0.112, 0.1146, 0.1174]

        radial_noise = [0.0108, 0.0119, 0.0136, 0.0156, 0.0178, 0.0202, 0.0226, 0.0251, 0.0277, 0.0303, 0.0329, 0.0355, 0.0383, 0.0408, 0.0435, 0.0461, 0.0488, 0.0515, 0.0541, 0.0568, 0.0596, 0.0622, 0.065, 0.0674, 0.0701, 0.0727, 0.0753, 0.0782, 0.0809, 0.0833, 0.086, 0.0883, 0.0914, 0.0937, 0.0966, 0.099, 0.1017, 0.1044, 0.1072, 0.1093]

        spiral_noise = [0.0077, 0.0096, 0.012, 0.0147, 0.0175, 0.0204, 0.0234, 0.0263, 0.0292, 0.0322, 0.0352, 0.0382, 0.0411, 0.0441, 0.047, 0.0499, 0.0528, 0.0558, 0.0587, 0.0616, 0.0644, 0.0676, 0.0703, 0.0731, 0.0759, 0.0788, 0.0816, 0.0848, 0.0873, 0.0901, 0.093, 0.0958, 0.0985, 0.1015, 0.1042, 0.1071, 0.1098, 0.1123, 0.1152, 0.1178]

        absolute_noise = [0.0025, 0.0051, 0.0076, 0.0101, 0.0127, 0.0153, 0.0178, 0.0203, 0.0229, 0.0254, 0.0279, 0.0305, 0.033, 0.0355, 0.0382, 0.0407, 0.0431, 0.0457, 0.0482, 0.0508, 0.0533, 0.0557, 0.0583, 0.061, 0.0635, 0.066, 0.0686, 0.0711, 0.0737, 0.0762, 0.0787, 0.0811, 0.0837, 0.0862, 0.0888, 0.0913, 0.0937, 0.0967, 0.0992, 0.1015]

        median = 0.0187

        std = 0.0038

    if sigma == 1.5 :
        plane_noise = [0.0107, 0.0123, 0.0146, 0.0172, 0.0201, 0.0231, 0.0261, 0.0292, 0.0323, 0.0355, 0.0387, 0.0418, 0.0449, 0.0482, 0.0513, 0.0544, 0.0576, 0.0609, 0.0639, 0.067, 0.0701, 0.0732, 0.0764, 0.0794, 0.0824, 0.0855, 0.0885, 0.0917, 0.0948, 0.0979, 0.1009, 0.1039, 0.1069, 0.1098, 0.1128, 0.1159, 0.1188, 0.1219, 0.1245, 0.1278]

        radial_noise = [0.0217, 0.0224, 0.0235, 0.0249, 0.0266, 0.0285, 0.0306, 0.0328, 0.0352, 0.0375, 0.0401, 0.0427, 0.0452, 0.048, 0.0506, 0.0532, 0.056, 0.0586, 0.0615, 0.0643, 0.067, 0.0698, 0.0726, 0.0754, 0.0781, 0.081, 0.0838, 0.0865, 0.0891, 0.0921, 0.0949, 0.0977, 0.1002, 0.1034, 0.1061, 0.1089, 0.1117, 0.1143, 0.1173, 0.1201]

        spiral_noise = [0.0131, 0.0145, 0.0165, 0.0189, 0.0215, 0.0244, 0.0273, 0.0303, 0.0334, 0.0364, 0.0395, 0.0427, 0.0457, 0.0488, 0.0521, 0.0551, 0.0584, 0.0614, 0.0644, 0.0676, 0.0708, 0.0738, 0.077, 0.0801, 0.0833, 0.0862, 0.0893, 0.0922, 0.0954, 0.0983, 0.1016, 0.1043, 0.1075, 0.1105, 0.1134, 0.1163, 0.1193, 0.1225, 0.1251, 0.1283]

        absolute_noise = [0.0028, 0.0055, 0.0083, 0.011, 0.0138, 0.0166, 0.0193, 0.022, 0.0248, 0.0276, 0.0303, 0.033, 0.0357, 0.0385, 0.0413, 0.0441, 0.0468, 0.0497, 0.0523, 0.0549, 0.0578, 0.0606, 0.0631, 0.0665, 0.0688, 0.0716, 0.0742, 0.0771, 0.0799, 0.0827, 0.0854, 0.0881, 0.0906, 0.0937, 0.0962, 0.0993, 0.1022, 0.1047, 0.1072, 0.1099]

        median = 0.0235

        std = 0.0055

    if sigma == 2 :

        plane_noise = [0.0174, 0.0185, 0.0201, 0.0222, 0.0247, 0.0272, 0.03, 0.0329, 0.0359, 0.0388, 0.0419, 0.045, 0.0482, 0.0512, 0.0543, 0.0574, 0.0606, 0.0638, 0.067, 0.07, 0.0733, 0.0765, 0.0795, 0.0826, 0.0857, 0.0889, 0.0919, 0.095, 0.098, 0.1013, 0.1043, 0.1073, 0.1106, 0.1135, 0.1163, 0.1196, 0.1225, 0.1257, 0.1287, 0.1315]

        radial_noise = [0.0348, 0.0352, 0.0359, 0.0369, 0.0381, 0.0396, 0.0411, 0.0429, 0.0448, 0.0468, 0.0489, 0.0512, 0.0534, 0.0558, 0.0582, 0.0606, 0.0631, 0.0657, 0.0683, 0.0709, 0.0736, 0.0761, 0.0788, 0.0816, 0.0841, 0.0869, 0.0897, 0.0923, 0.095, 0.098, 0.1005, 0.1033, 0.1063, 0.109, 0.1114, 0.1144, 0.1173, 0.12, 0.1228, 0.1256]

        spiral_noise = [0.02, 0.021, 0.0225, 0.0244, 0.0266, 0.029, 0.0317, 0.0344, 0.0373, 0.0402, 0.0432, 0.0462, 0.0493, 0.0524, 0.0554, 0.0584, 0.0616, 0.0647, 0.0678, 0.0709, 0.074, 0.0772, 0.0802, 0.0835, 0.0865, 0.0894, 0.0926, 0.0958, 0.0989, 0.1023, 0.1051, 0.108, 0.1112, 0.1144, 0.1174, 0.1204, 0.1233, 0.1262, 0.1296, 0.1322]

        absolute_noise = [0.0028, 0.0056, 0.0085, 0.0113, 0.0141, 0.017, 0.0198, 0.0226, 0.0254, 0.0281, 0.0311, 0.0339, 0.0367, 0.0396, 0.0425, 0.0452, 0.048, 0.051, 0.0537, 0.0565, 0.0594, 0.0623, 0.0649, 0.0678, 0.0705, 0.0736, 0.0762, 0.0793, 0.082, 0.085, 0.0875, 0.0903, 0.0931, 0.0959, 0.0989, 0.1018, 0.1044, 0.1073, 0.1105, 0.1132]

        median = 0.0273

        std = 0.007

    if sigma == 3 :

        plane_noise = [0.0341, 0.0347, 0.0356, 0.0369, 0.0385, 0.0403, 0.0422, 0.0444, 0.0467, 0.0492, 0.0517, 0.0543, 0.057, 0.0598, 0.0624, 0.0653, 0.0683, 0.0712, 0.074, 0.0768, 0.08, 0.0827, 0.0858, 0.0889, 0.0919, 0.095, 0.0977, 0.1008, 0.1039, 0.1069, 0.1097, 0.1129, 0.1158, 0.1188, 0.1219, 0.1248, 0.1278, 0.1309, 0.1337, 0.1366]

        radial_noise = [0.0618, 0.0621, 0.0625, 0.063, 0.0637, 0.0646, 0.0656, 0.0667, 0.0679, 0.0693, 0.0707, 0.0722, 0.0739, 0.0756, 0.0773, 0.0792, 0.0811, 0.083, 0.0853, 0.0873, 0.0893, 0.0915, 0.0938, 0.0959, 0.0984, 0.1008, 0.1031, 0.1055, 0.1077, 0.1102, 0.1129, 0.115, 0.1181, 0.1205, 0.1227, 0.1252, 0.1279, 0.1306, 0.1331, 0.1354]

        spiral_noise = [0.0361, 0.0367, 0.0376, 0.0389, 0.0404, 0.0421, 0.044, 0.0461, 0.0483, 0.0507, 0.0531, 0.0556, 0.0582, 0.0609, 0.0636, 0.0663, 0.0692, 0.0721, 0.0749, 0.0778, 0.0807, 0.0837, 0.0867, 0.0897, 0.0925, 0.0956, 0.0987, 0.1016, 0.1044, 0.1076, 0.1105, 0.1136, 0.1164, 0.1196, 0.1225, 0.1252, 0.1284, 0.1314, 0.1346, 0.1376]

        absolute_noise = [0.0029, 0.0058, 0.0086, 0.0115, 0.0144, 0.0172, 0.0201, 0.0231, 0.0259, 0.0288, 0.0316, 0.0346, 0.0374, 0.0402, 0.0432, 0.0462, 0.049, 0.0518, 0.0545, 0.0575, 0.0605, 0.0633, 0.0661, 0.0693, 0.0717, 0.0749, 0.0778, 0.0805, 0.0834, 0.0865, 0.0891, 0.0924, 0.0946, 0.0977, 0.1009, 0.1033, 0.1066, 0.1091, 0.1122, 0.1154]

        median = 0.0339
        std = 0.0094

    if sigma == 4 :

        plane_noise = [0.0526, 0.053, 0.0536, 0.0545, 0.0556, 0.0569, 0.0583, 0.0599, 0.0617, 0.0635, 0.0655, 0.0676, 0.0697, 0.072, 0.0743, 0.0768, 0.0792, 0.0817, 0.0842, 0.0867, 0.0896, 0.0921, 0.0948, 0.0975, 0.1003, 0.1031, 0.1057, 0.1084, 0.1114, 0.1141, 0.1167, 0.1196, 0.1228, 0.1255, 0.1285, 0.1313, 0.134, 0.1365, 0.1395, 0.1424]

        radial_noise = [0.0858, 0.086, 0.0863, 0.0867, 0.0872, 0.0877, 0.0884, 0.0893, 0.09, 0.091, 0.0919, 0.093, 0.0943, 0.0956, 0.0969, 0.0983, 0.0999, 0.1013, 0.1029, 0.1045, 0.1063, 0.1082, 0.11, 0.1118, 0.1139, 0.1156, 0.1176, 0.1197, 0.1217, 0.1238, 0.1258, 0.1279, 0.1302, 0.1327, 0.1348, 0.1369, 0.1394, 0.1414, 0.1439, 0.1462]

        spiral_noise = [0.0532, 0.0536, 0.0542, 0.0551, 0.0562, 0.0574, 0.0588, 0.0604, 0.0621, 0.0639, 0.0658, 0.0679, 0.07, 0.0722, 0.0745, 0.0769, 0.0794, 0.0818, 0.0844, 0.0869, 0.0894, 0.0921, 0.0949, 0.0975, 0.1002, 0.1031, 0.1057, 0.1085, 0.1111, 0.114, 0.1171, 0.1197, 0.1226, 0.1252, 0.1283, 0.1312, 0.134, 0.1369, 0.1397, 0.1423]

        absolute_noise = [0.0029, 0.0058, 0.0087, 0.0116, 0.0145, 0.0174, 0.0202, 0.0231, 0.0261, 0.029, 0.0319, 0.0348, 0.0376, 0.0406, 0.0435, 0.0462, 0.0491, 0.0522, 0.0551, 0.0579, 0.0608, 0.0635, 0.0667, 0.0696, 0.0724, 0.0754, 0.0782, 0.0809, 0.0837, 0.0868, 0.0896, 0.0925, 0.0954, 0.0983, 0.1016, 0.1039, 0.1071, 0.11, 0.1128, 0.1161]

        median = 0.0397

        std = 0.011



    return plane_noise , radial_noise , spiral_noise , absolute_noise , median , std


sigma = 2
plane_noise , radial_noise , spiral_noise , absolute_noise , median , std = gaussian_sigma(sigma = sigma)

sequence = [0.005 * i for i in range(1, 41)]  # Generates first 20 terms


xaxis = [i for i in range(0, 21)]  # Generates first 20 terms



plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42

figsize_cm = (9, 4.5)  # example in cm
figsize_in = tuple(x / 2.54 for x in figsize_cm)  # convert to inches

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_in)
#fig.subplots_adjust(wspace=0.5)  # spacing between ax1 and ax2


tick_positions = np.linspace(0, 21, 21)  # positions: 1 to 41
tick_labels_full = np.linspace(0.005, 0.205, len(tick_positions))  # labels: 0.005 to 0.205

desired_labels_main = [0,0.05, 0.10,0.15, 0.20]
tick_positions_filtered = [label / 0.005 for label in desired_labels_main]
tick_labels_filtered = [f"{label:.2f}" for label in desired_labels_main]
tick_labels_filtered[0]='0'
tick_labels_filtered[1]='0.1'
tick_labels_filtered[2]='0.2'

ax1.set_xticks(tick_positions_filtered)
ax1.set_xticklabels(tick_labels_filtered)


ax1.plot(xaxis,plane_spatial , label='Plane' , color = 'royalblue' ,linewidth=2)
ax1.plot(xaxis,radial_spatial , label='Radial' , color ='tomato' ,linewidth=2, linestyle=(0, (3, 3)))
ax1.plot(xaxis,two_gaussians_spatial , label='2 Gauss' , color = 'mediumseagreen' ,linewidth=2)


ax1.set_ylim([0,1.05])
ax1.set_xlim([0,20])
#ax1.set_title(fr'Score vs Noise Spatialy Gaussian smooth $\sigma$ = {sigma}', fontsize=12)
ax1.set_xlabel('Std Noise added ', fontsize=10)
ax1.set_ylabel('Score', fontsize=10)


# Your numbers
bin_size = 0.01

# Convert to x-tick index if your x-axis is index-based
median_idx = median / bin_size
print(median)
std_left_idx = (median - std) / bin_size
std_right_idx = (median + std) / bin_size
ax1.axvline(x=(median / 0.01), color='darkblue', linestyle='--',label='Cortex\nNoise',linewidth=1) ## Convert to corresponding x tick

ax1.axvspan(std_left_idx, std_right_idx, color='lightsteelblue', alpha=0.5)#, label='Std range')


ax1.legend(
    frameon=False,       # removes the legend box
    loc='upper right',   # position (e.g., 'best', 'lower left', etc.)
    fontsize=5,         # font size
    ncol=1,              # number of columns
    handlelength=2     # length of the legend line
)

x_min, x_max = ax1.get_xlim()
y_min, y_max = ax1.get_ylim()
ax1.set_aspect((x_max - x_min) / (y_max - y_min), adjustable='box')

ax1.tick_params(axis='x', labelsize=8)
ax1.tick_params(axis='y', labelsize=8)

#ax11 = inset_axes(ax1, width="50%", height="50%", bbox_to_anchor=(0.3, 0.15, 0.5, 0.5), bbox_transform=ax1.transAxes, loc='lower left')
#ax11.plot(xaxis,plane_noise , label='Plane' , color ='royalblue' ,linewidth=2)
#ax11.plot(xaxis,radial_noise , label='Radial' , color ='tomato' ,linewidth=2)
#ax11.plot(xaxis,spiral_noise , label='Spiral' , color = 'mediumseagreen' ,linewidth=2,linestyle=(0, (3, 3)))
#ax2.plot(xaxis,absolute_noise , color = 'maroon',linewidth=2)

desired_labels_second = [0, 0.1, 0.2]
tick_positions_filtered = [label / 0.01 for label in desired_labels_second]
tick_labels_filtered = [f"{label:.1f}" for label in desired_labels_second]








def data_sigma_1():
    #### 54MRL
    Spatial_54MRL = [0.014, 0.0148, 0.0144, 0.0225, 0.0197, 0.02, 0.0137, 0.0181, 0.019, 0.026, 0.0235, 0.0182, 0.019, 0.0219, 0.0162, 0.0221, 0.0229, 0.0138, 0.0208, 0.0165, 0.0161, 0.0147, 0.0177, 0.0159, 0.0137, 0.0161, 0.0285, 0.0251, 0.0182, 0.0156, 0.0185, 0.0287, 0.0177, 0.0201, 0.0151, 0.018, 0.0333, 0.0203, 0.0171, 0.02, 0.0158, 0.0204, 0.0194, 0.0254, 0.0171, 0.0158, 0.0155, 0.0175, 0.0151, 0.0203, 0.0202, 0.0331, 0.0165, 0.0215, 0.0219, 0.0148, 0.015, 0.0162, 0.0213, 0.0195, 0.016, 0.0162, 0.0219, 0.0198, 0.0222, 0.0166, 0.019, 0.0201, 0.0181, 0.0195, 0.0212, 0.0182, 0.0274, 0.025, 0.0208, 0.0246, 0.0238, 0.0229, 0.0256, 0.019, 0.02, 0.0183, 0.0246, 0.0198, 0.0208, 0.0195, 0.0307, 0.0213, 0.0178, 0.0202, 0.0215, 0.0194, 0.0157, 0.024, 0.0199, 0.0274, 0.0252, 0.0209, 0.0182, 0.0191, 0.0263, 0.02, 0.0204, 0.0157, 0.0257, 0.0235, 0.0238, 0.0174, 0.0205, 0.0163, 0.0163, 0.017, 0.0239, 0.0169, 0.03, 0.017, 0.0159, 0.0166, 0.0208, 0.0178, 0.0162, 0.0161, 0.0154, 0.026, 0.0175, 0.0177, 0.0198, 0.0155, 0.0233, 0.0165, 0.0236, 0.0169, 0.0163, 0.0168, 0.0248, 0.0276, 0.017, 0.0156, 0.0193, 0.0273, 0.0211, 0.0171, 0.0174, 0.0245, 0.0236, 0.0154, 0.0206, 0.0194, 0.0203, 0.0249, 0.0166, 0.0166, 0.0185, 0.0167, 0.0234, 0.0151, 0.0167, 0.02, 0.017, 0.0153, 0.0177, 0.0253, 0.0238, 0.0143, 0.0169, 0.0171, 0.0168, 0.0238, 0.0278, 0.0186, 0.0211, 0.0313, 0.0176, 0.0177, 0.0159, 0.023, 0.0194, 0.0284, 0.018, 0.0189, 0.0153, 0.0157, 0.0278, 0.0172, 0.0159, 0.0159, 0.0153, 0.0147, 0.0306, 0.025, 0.0227, 0.0161, 0.0155, 0.0159, 0.019, 0.0212, 0.0173, 0.0206, 0.0156, 0.0246, 0.0211, 0.0238, 0.0322, 0.0217, 0.0146, 0.016, 0.0193, 0.0145, 0.0329, 0.0202, 0.0148, 0.0155, 0.0166, 0.015, 0.0287, 0.0156, 0.0149, 0.0145, 0.0142, 0.0143, 0.0196, 0.0193, 0.0165, 0.026, 0.0178, 0.0189, 0.0208, 0.0279, 0.0233, 0.0266, 0.0309, 0.0168, 0.0314, 0.0176, 0.0205, 0.0169, 0.0304, 0.0168, 0.0161, 0.0161, 0.017, 0.0196, 0.0171, 0.0189, 0.0188, 0.0191, 0.0183, 0.0203, 0.018, 0.0195, 0.0254, 0.0262, 0.017, 0.0163, 0.0229, 0.0192, 0.0255, 0.0182, 0.0169, 0.0174, 0.0172, 0.0154, 0.0186, 0.0151, 0.0149, 0.0147, 0.0189, 0.0183, 0.0143, 0.0153, 0.0146, 0.0166, 0.0145, 0.0141, 0.0182, 0.0156, 0.0151, 0.0198, 0.0174]

    #### 63MR
    Spatial_63MR = [0.0169, 0.0126, 0.0126, 0.0133, 0.017, 0.0204, 0.0163, 0.0125, 0.0167, 0.018, 0.0129, 0.0141, 0.0159, 0.0131, 0.0171, 0.015, 0.0225, 0.0131, 0.013, 0.0222, 0.0143, 0.0143, 0.0139, 0.0177, 0.0131, 0.0166, 0.0133, 0.013, 0.0193, 0.0205, 0.0127, 0.015, 0.0165, 0.0172, 0.0182, 0.0165, 0.0238, 0.0157, 0.0184, 0.0218, 0.0188, 0.0171, 0.0159, 0.0139, 0.0159, 0.0153, 0.0156, 0.0163, 0.0192, 0.0165, 0.015, 0.0146, 0.017, 0.0215, 0.0215, 0.0177, 0.0188, 0.0149, 0.0242, 0.0157, 0.0196, 0.0177, 0.0165, 0.0227, 0.0284, 0.0148, 0.0183, 0.0153, 0.016, 0.0175, 0.0177, 0.0181, 0.0146, 0.0217, 0.0215, 0.0134, 0.0239, 0.0161, 0.0151, 0.0167, 0.0218, 0.0164, 0.0157, 0.0139, 0.0125, 0.0118, 0.0126, 0.0196, 0.0133, 0.0121, 0.0134, 0.0275, 0.0221, 0.015, 0.0181, 0.0126, 0.0223, 0.0155, 0.0182, 0.0281, 0.0176, 0.0203, 0.0177, 0.0237, 0.0179, 0.0194, 0.0202, 0.0239, 0.018, 0.0181, 0.0193, 0.0134, 0.0144, 0.0161, 0.0158, 0.0134, 0.0155, 0.0134, 0.0167, 0.0188, 0.0201, 0.0173, 0.0144, 0.0205, 0.0282, 0.0335, 0.0124, 0.0134, 0.013, 0.0119, 0.0131, 0.0149, 0.0189, 0.024, 0.0215, 0.0143, 0.0119, 0.0207, 0.0108, 0.0255, 0.0172, 0.0141, 0.0248, 0.0285, 0.0149, 0.0167, 0.0137, 0.0281, 0.016, 0.0227, 0.016, 0.0198, 0.0239, 0.0143, 0.0163, 0.0144, 0.0195, 0.0225, 0.0218, 0.0248, 0.0207, 0.0135, 0.0146, 0.014, 0.0153, 0.0166, 0.0147, 0.0285, 0.0167, 0.0141, 0.0238, 0.0142, 0.0162, 0.0155, 0.0174, 0.0195, 0.0141, 0.0169, 0.0363, 0.0147, 0.0151, 0.0168, 0.0163, 0.0155, 0.0185, 0.0148, 0.0268, 0.0193, 0.016, 0.0171, 0.0147, 0.0146, 0.0154, 0.0151, 0.0161, 0.0193, 0.0255, 0.0161, 0.0274, 0.0176, 0.0183, 0.01, 0.0234, 0.0211, 0.0215, 0.0277, 0.0138, 0.0111, 0.0204, 0.0216, 0.017]

    #### 187FN
    Spatial_187FN = [0.016, 0.0149, 0.016, 0.027, 0.0201, 0.0272, 0.015, 0.0144, 0.0142, 0.0154, 0.0145, 0.0144, 0.0138, 0.0151, 0.0153, 0.0158, 0.0146, 0.0155, 0.015, 0.0154, 0.0151, 0.0159, 0.014, 0.0174, 0.0145, 0.0155, 0.0145, 0.0153, 0.0212, 0.0148, 0.0145, 0.0154, 0.0166, 0.0136, 0.0148, 0.015, 0.0152, 0.0155, 0.0143, 0.015, 0.0159, 0.0141, 0.0146, 0.0143, 0.0206, 0.0229, 0.0201, 0.025, 0.022, 0.0253, 0.0209, 0.0181, 0.0236, 0.0209, 0.0199, 0.0246, 0.0196, 0.0216, 0.0219, 0.0208, 0.0192, 0.0198, 0.018, 0.0209, 0.0193, 0.0193, 0.0194, 0.0216, 0.0191, 0.0177, 0.0229, 0.0224, 0.0193, 0.0215, 0.0211, 0.0229, 0.0227, 0.0211, 0.0205, 0.0204, 0.0228, 0.0203, 0.0186, 0.0205, 0.0208, 0.0181, 0.0234, 0.0217, 0.0233, 0.0317, 0.021, 0.0216, 0.0204, 0.0191, 0.0265, 0.0225, 0.0226, 0.0228, 0.0264, 0.0219, 0.0254, 0.015, 0.0172, 0.0149, 0.0156, 0.0174, 0.0189, 0.0142, 0.0139, 0.0144, 0.0153, 0.0146, 0.0169, 0.0151, 0.015, 0.0163, 0.0156, 0.0214, 0.0132, 0.0146, 0.0169, 0.0181, 0.0147, 0.018, 0.0156, 0.0192, 0.0186, 0.0151, 0.0187, 0.0343, 0.0153, 0.0146, 0.015, 0.0138, 0.0127, 0.0169, 0.016, 0.0188, 0.0155, 0.0209, 0.0181, 0.019, 0.0143, 0.0165, 0.0194, 0.017, 0.0181, 0.0197, 0.0196, 0.0174, 0.0148, 0.0152, 0.0151, 0.0197, 0.0184, 0.0175, 0.0204, 0.0197, 0.0195, 0.0171, 0.0178, 0.0153, 0.0192, 0.0168, 0.0197, 0.0189, 0.0203, 0.0212, 0.0176, 0.0222, 0.0217, 0.0167, 0.0262, 0.0159, 0.0187, 0.0178, 0.0178, 0.0173, 0.0181, 0.0236, 0.0241, 0.0211, 0.0195, 0.0223, 0.0224, 0.0181, 0.0239, 0.0177, 0.0218, 0.0229, 0.016, 0.0171, 0.0193, 0.0147, 0.0254, 0.0183, 0.0249, 0.0184, 0.017, 0.0169, 0.0198, 0.0171, 0.0163, 0.0188, 0.033, 0.0185, 0.0176, 0.0191, 0.0229, 0.0177, 0.0175, 0.0201, 0.0147, 0.0175, 0.0178, 0.0183, 0.0161, 0.0165, 0.0168, 0.0164, 0.0176, 0.0182, 0.0159, 0.0154, 0.0221, 0.0179, 0.0198, 0.0185, 0.0156, 0.018, 0.0184, 0.0188, 0.0212, 0.0202, 0.0216, 0.0159, 0.0177, 0.0168, 0.0201, 0.0164, 0.0157, 0.0159, 0.0173, 0.0155, 0.0152, 0.017, 0.0208, 0.0191, 0.0214, 0.019, 0.0182, 0.0153, 0.016, 0.0325, 0.0164, 0.0156, 0.0168, 0.0182, 0.0176, 0.0159, 0.0162, 0.014, 0.0174, 0.0202, 0.0143, 0.0215, 0.0183, 0.0178, 0.0206, 0.02, 0.0198, 0.0272, 0.0241, 0.0196, 0.0189, 0.0197, 0.0175, 0.0182, 0.0215, 0.0174, 0.0162, 0.0196, 0.0176, 0.0196, 0.0204, 0.0203, 0.0205, 0.0193, 0.0181, 0.0182, 0.0193, 0.0193, 0.0213, 0.0305, 0.0172, 0.0185, 0.0166, 0.0236, 0.0186, 0.0175, 0.0206, 0.019, 0.0214, 0.0199, 0.0239, 0.0213, 0.0211, 0.0189, 0.0232, 0.0217, 0.0195, 0.0262, 0.0218, 0.0217, 0.0173, 0.019, 0.026, 0.032, 0.0203, 0.016, 0.0138, 0.0217, 0.0148, 0.0143, 0.0151, 0.0155, 0.0149, 0.0139, 0.0177, 0.0146, 0.0192, 0.0135, 0.0154, 0.0149, 0.0159, 0.0245, 0.015, 0.0162, 0.0253, 0.0184, 0.0138, 0.0172, 0.0165, 0.0165, 0.0174, 0.019, 0.0218, 0.0241, 0.017, 0.0259, 0.0179, 0.031, 0.0137, 0.0158, 0.0145, 0.0153, 0.0147, 0.0144, 0.0152, 0.0138, 0.0158, 0.03, 0.027, 0.0204, 0.0232, 0.0147, 0.0131, 0.0178, 0.0156, 0.0134, 0.0154, 0.0138, 0.0235, 0.0142, 0.0142, 0.0326, 0.0168, 0.0145, 0.0162, 0.0135, 0.0145, 0.0147, 0.0169, 0.0148, 0.0153]

    #### 203MN
    Spatial_203MN = [0.0163, 0.017, 0.0179, 0.0153, 0.0161, 0.0163, 0.0166, 0.0163, 0.0183, 0.017, 0.016, 0.0178, 0.0184, 0.0164, 0.0246, 0.0187, 0.0284, 0.0168, 0.0162, 0.0166, 0.0178, 0.0179, 0.0165, 0.016, 0.019, 0.017, 0.0168, 0.0167, 0.017, 0.0165, 0.0167, 0.0171, 0.0166, 0.0169, 0.0243, 0.0213, 0.0214, 0.0219, 0.0231, 0.021, 0.0222, 0.0227, 0.0217, 0.0221, 0.0218, 0.0231, 0.0217, 0.0218, 0.0229, 0.0229, 0.022, 0.0212, 0.0233, 0.0224, 0.0223, 0.0212, 0.021, 0.0213, 0.023, 0.0222, 0.0222, 0.0221, 0.0216, 0.0317, 0.0232, 0.0222, 0.0208, 0.0212, 0.0216, 0.0222, 0.0224, 0.0223, 0.0222, 0.0221, 0.0225, 0.0231, 0.0222, 0.0214, 0.0246, 0.0208, 0.0222, 0.0222, 0.0212, 0.0272, 0.0208, 0.0223, 0.0207, 0.0214, 0.022, 0.0213, 0.0159, 0.0156, 0.016, 0.0163, 0.0165, 0.0197, 0.0164, 0.0166, 0.0161, 0.0163, 0.0165, 0.0165, 0.0189, 0.0309, 0.0154, 0.0155, 0.0184, 0.0147, 0.0176, 0.0167, 0.0163, 0.0162, 0.0157, 0.0169, 0.0156, 0.0157, 0.0172, 0.0185, 0.0164, 0.0161, 0.0166, 0.0186, 0.0167, 0.0165, 0.0157, 0.0156, 0.0296, 0.0238, 0.0183, 0.0157, 0.0158, 0.0166, 0.0169, 0.0186, 0.0195, 0.0198, 0.0197, 0.0195, 0.0197, 0.0201, 0.0189, 0.0197, 0.0194, 0.0179, 0.0198, 0.021, 0.02, 0.0199, 0.0208, 0.02, 0.0187, 0.0201, 0.019, 0.0199, 0.0187, 0.0223, 0.0195, 0.019, 0.0214, 0.0211, 0.0211, 0.0193, 0.0205, 0.0184, 0.0336, 0.037, 0.0201, 0.0177, 0.0193, 0.0192, 0.0186, 0.0198, 0.0194, 0.018, 0.0204, 0.0192, 0.0219, 0.0188, 0.0198, 0.0185, 0.0199, 0.0205, 0.0174, 0.0181, 0.0196, 0.0237, 0.0201, 0.0201, 0.0286, 0.0174, 0.0211, 0.0191, 0.017, 0.0175, 0.0175, 0.0222, 0.0193, 0.0178, 0.0181, 0.0185, 0.0178, 0.0184, 0.0181, 0.0181, 0.019, 0.0184, 0.0181, 0.0212, 0.0167, 0.0177, 0.018, 0.0181, 0.0186, 0.018, 0.0193, 0.0183, 0.0203, 0.0201, 0.0193, 0.0181, 0.0195, 0.0198, 0.026, 0.018, 0.0205, 0.017, 0.0174, 0.018, 0.0185, 0.0171, 0.0158, 0.0156, 0.0178, 0.0169, 0.0176, 0.0159, 0.0191, 0.0185, 0.016, 0.0157, 0.0153, 0.0155, 0.0168, 0.0161, 0.017, 0.0246, 0.0165, 0.0155, 0.0158, 0.0161, 0.0158, 0.0151, 0.0159, 0.0154, 0.0165, 0.0205, 0.0162, 0.0159, 0.0327, 0.0217, 0.0268, 0.0194, 0.017, 0.0147, 0.0155, 0.0166, 0.0177, 0.016, 0.0176, 0.0162, 0.0182, 0.0165, 0.0158, 0.0262, 0.0218, 0.017, 0.0165, 0.018, 0.0162, 0.0171, 0.0189, 0.0188, 0.019, 0.018, 0.0185, 0.0191, 0.0165, 0.0164, 0.0181, 0.0151, 0.0197, 0.0176, 0.0198, 0.0275, 0.0171, 0.0172, 0.0163, 0.0164, 0.0218, 0.0172, 0.0212, 0.0225, 0.0334, 0.0205, 0.0162, 0.015, 0.0171, 0.0155, 0.0189, 0.0178, 0.0174, 0.0181, 0.0152, 0.0168, 0.0189, 0.0168, 0.019, 0.0158, 0.0155, 0.0145, 0.0143, 0.03, 0.0146, 0.0156, 0.0159, 0.0166, 0.0164, 0.0155, 0.0163, 0.0166, 0.0172, 0.0234]

    #### 204FR
    Spatial_204FR = [0.0139, 0.0285, 0.0145, 0.023, 0.0262, 0.0197, 0.0133, 0.0136, 0.0136, 0.0142, 0.0148, 0.0135, 0.0248, 0.0159, 0.0161, 0.0169, 0.0161, 0.0282, 0.0224, 0.0244, 0.0269, 0.019, 0.0178, 0.0157, 0.0152, 0.0153, 0.0167, 0.0173, 0.0149, 0.0163, 0.0298, 0.0239, 0.0301, 0.024, 0.0165, 0.0159, 0.0158, 0.0164, 0.0167, 0.0164, 0.0158, 0.0208, 0.016, 0.0154, 0.0159, 0.0188, 0.0152, 0.0189, 0.0182, 0.019, 0.0192, 0.0185, 0.0197, 0.0193, 0.0208, 0.017, 0.0184, 0.0178, 0.0178, 0.0313, 0.0193, 0.0188, 0.0186, 0.02, 0.0198, 0.0192, 0.0181, 0.0172, 0.0201, 0.0196, 0.0191, 0.0195, 0.02, 0.0228, 0.0194, 0.0195, 0.0346, 0.0296, 0.0254, 0.0181, 0.0182, 0.0193, 0.0188, 0.019, 0.0245, 0.0187, 0.0185, 0.0184, 0.0198, 0.0198, 0.0194, 0.021, 0.0188, 0.0185, 0.018, 0.0287, 0.0212, 0.0169, 0.0178, 0.0193, 0.0263, 0.0257, 0.0269, 0.0263, 0.0258, 0.0269, 0.0259, 0.0261, 0.0274, 0.0257, 0.0254, 0.0269, 0.0265, 0.026, 0.0263, 0.0256, 0.0263, 0.0257, 0.0248, 0.0248, 0.0265, 0.0258, 0.0267, 0.0259, 0.0248, 0.0263, 0.0261, 0.0262, 0.0247, 0.0265, 0.0203, 0.0192, 0.0203, 0.0201, 0.0201, 0.019, 0.0194, 0.0194, 0.0181, 0.019, 0.0184, 0.0181, 0.0178, 0.0208, 0.0222, 0.0242, 0.0205, 0.0193, 0.0199, 0.0194, 0.0197, 0.0186, 0.0194, 0.0199, 0.0376, 0.0317, 0.0185, 0.0296, 0.0191, 0.0189, 0.0189, 0.0189, 0.0205, 0.0199, 0.0202, 0.0187, 0.0193, 0.0197, 0.0194, 0.02, 0.0194, 0.0205, 0.0218, 0.019, 0.0189, 0.0185, 0.021, 0.0198, 0.0261, 0.0285, 0.0223, 0.0202, 0.0199, 0.0195, 0.0217, 0.0209, 0.0202, 0.0214, 0.0209, 0.0205, 0.0205, 0.0204, 0.0205, 0.0209, 0.02, 0.0214, 0.0207, 0.0194, 0.0198, 0.019, 0.0203, 0.019, 0.0223, 0.0203, 0.0199, 0.0181, 0.0202, 0.0204, 0.0284, 0.022, 0.0222, 0.0221, 0.0206, 0.019, 0.0233, 0.0198, 0.0223, 0.0253, 0.0186, 0.0197, 0.02, 0.0204, 0.0204, 0.0217, 0.0217, 0.0204, 0.0202, 0.0198, 0.0204, 0.0208, 0.0213, 0.0209, 0.0206, 0.0211, 0.0201, 0.0211, 0.0219, 0.0229, 0.0198, 0.021, 0.0224, 0.02, 0.0196, 0.0203, 0.021, 0.0202, 0.0196, 0.0199, 0.0236, 0.024, 0.0367, 0.0309, 0.02, 0.0192, 0.0192, 0.0194, 0.0199, 0.0218, 0.0205, 0.0225, 0.0213, 0.025, 0.0266, 0.0194, 0.0241, 0.0206, 0.0204, 0.0203, 0.0194, 0.0197, 0.0215, 0.0171, 0.0228, 0.0228, 0.0224, 0.0181, 0.0169, 0.0165, 0.0166, 0.0182, 0.0258, 0.0263, 0.0215, 0.0209, 0.0181, 0.0182, 0.0184, 0.0183]

    #### 206FRL
    Spatial_206FRL = [0.0173, 0.0171, 0.0188, 0.0179, 0.0187, 0.0162, 0.0176, 0.0198, 0.0166, 0.017, 0.0169, 0.0169, 0.0167, 0.0172, 0.019, 0.0166, 0.0168, 0.017, 0.0161, 0.017, 0.017, 0.0173, 0.0184, 0.0171, 0.0161, 0.0171, 0.0171, 0.0166, 0.0164, 0.0231, 0.0205, 0.0162, 0.017, 0.0182, 0.018, 0.0168, 0.0165, 0.0165, 0.0153, 0.0179, 0.016, 0.0149, 0.0176, 0.0165, 0.015, 0.0153, 0.0151, 0.0159, 0.0139, 0.0146, 0.0156, 0.015, 0.0148, 0.0198, 0.0205, 0.0154, 0.025, 0.0207, 0.0142, 0.0149, 0.0167, 0.0166, 0.0249, 0.0178, 0.0159, 0.0152, 0.0157, 0.0163, 0.0156, 0.0154, 0.0158, 0.0164, 0.0153, 0.0155, 0.0156, 0.0193, 0.0157, 0.0169, 0.0151, 0.0159, 0.0151, 0.0156, 0.0154, 0.0165, 0.0184, 0.0145, 0.0154, 0.0266, 0.0236, 0.0242, 0.0221, 0.0233, 0.0149, 0.0162, 0.0155, 0.0162, 0.0161, 0.0144, 0.0146, 0.0166, 0.0163, 0.0151, 0.017, 0.0154, 0.0166, 0.022, 0.0164, 0.0162, 0.0153, 0.0159, 0.0158, 0.0158, 0.0144, 0.0164, 0.0269, 0.0231, 0.0203, 0.0206, 0.02, 0.018, 0.0211, 0.021, 0.0203, 0.0204, 0.0203, 0.0211, 0.02, 0.0229, 0.0219, 0.0216, 0.0201, 0.0217, 0.02, 0.0204, 0.0214, 0.0211, 0.0208, 0.0257, 0.0198, 0.0204, 0.0201, 0.0203, 0.02, 0.0205, 0.0208, 0.021, 0.0216, 0.0201, 0.0205, 0.0203, 0.0206, 0.0201, 0.0203, 0.0202, 0.0198, 0.0209, 0.0189, 0.0205, 0.0194, 0.0201, 0.0181, 0.0188, 0.0182, 0.0179, 0.018, 0.0192, 0.0177, 0.0182, 0.0191, 0.0216, 0.0171, 0.0178, 0.0179, 0.0184, 0.0186, 0.0177, 0.0182, 0.0183, 0.0183, 0.0184, 0.0181, 0.0182, 0.0187, 0.0183, 0.0186, 0.018, 0.019, 0.0176, 0.0182, 0.0181, 0.0233, 0.0207, 0.0203, 0.0173, 0.0178, 0.0182, 0.0183, 0.0302, 0.0215, 0.0217, 0.0193, 0.018, 0.0177, 0.0181, 0.018, 0.0185, 0.0193, 0.018, 0.0165, 0.0189, 0.0224, 0.018, 0.018, 0.02, 0.0198, 0.0193, 0.0191, 0.0192, 0.0222, 0.0188, 0.0194, 0.0196, 0.0197, 0.0195, 0.0198, 0.0202, 0.0199, 0.0198, 0.0193, 0.0194, 0.0209, 0.0206, 0.0208, 0.0217, 0.0231, 0.018, 0.0182, 0.0195, 0.0203, 0.0188, 0.0189, 0.0204, 0.0188, 0.0193, 0.0193, 0.0187, 0.0193, 0.019, 0.0187, 0.0186, 0.0191, 0.0182, 0.0187, 0.0189, 0.0297, 0.0265, 0.0221, 0.0194, 0.0186, 0.0197, 0.0185, 0.0168, 0.0211, 0.0203, 0.0192, 0.0195, 0.0189, 0.0256, 0.0181, 0.018, 0.019, 0.0205, 0.0201, 0.0199, 0.0201, 0.0192, 0.0195, 0.0202, 0.0188, 0.0187, 0.0187, 0.0198, 0.019, 0.0205, 0.0192, 0.0174, 0.0186, 0.0186, 0.02, 0.0191, 0.0197, 0.0186, 0.0186, 0.0239, 0.0253, 0.0182, 0.018, 0.0186, 0.0183, 0.0186, 0.0179, 0.0177, 0.0185, 0.0182, 0.0206, 0.022, 0.0202, 0.0201, 0.0191, 0.0189, 0.0186, 0.0194, 0.0196, 0.0189, 0.0192, 0.0188, 0.0188, 0.0255, 0.0256, 0.0267, 0.0265, 0.0271, 0.0267, 0.0295, 0.0288, 0.0272, 0.027, 0.0281, 0.0278, 0.03, 0.0272]


    #### 211MRR
    Spatial_211MRR = [0.0191, 0.0192, 0.0197, 0.0188, 0.019, 0.0185, 0.0198, 0.0197, 0.019, 0.0206, 0.0197, 0.0196, 0.0196, 0.0236, 0.0188, 0.0191, 0.0202, 0.0202, 0.0194, 0.0203, 0.0217, 0.0201, 0.0209, 0.023, 0.0221, 0.0197, 0.0223, 0.0207, 0.0216, 0.019, 0.0175, 0.0201, 0.0194, 0.019, 0.0185, 0.0189, 0.0194, 0.02, 0.0207, 0.0214, 0.0224, 0.02, 0.0205, 0.019, 0.0171, 0.0166, 0.0165, 0.0182, 0.0188, 0.0195, 0.0168, 0.0184, 0.0166, 0.019, 0.015, 0.0189, 0.0169, 0.0164, 0.017, 0.016, 0.0175, 0.0188, 0.0218, 0.0173, 0.0189, 0.0193, 0.0187, 0.0202, 0.0168, 0.0172, 0.0177, 0.0176, 0.0185, 0.0167, 0.0158, 0.0195, 0.0192, 0.0176, 0.0193, 0.0181, 0.0196, 0.02, 0.0197, 0.0156, 0.0199, 0.018, 0.0134, 0.013, 0.0136, 0.0153, 0.0159, 0.0184, 0.0134, 0.0131, 0.0139, 0.0154, 0.0137, 0.0137, 0.0134, 0.0146, 0.0145, 0.0137, 0.0149, 0.013, 0.0142, 0.0144, 0.015, 0.0146, 0.0133, 0.0136, 0.0129, 0.0137, 0.0141, 0.0134, 0.0174, 0.0252, 0.02, 0.0206, 0.0218, 0.0214, 0.017, 0.0204, 0.0196, 0.0197, 0.0169, 0.0221, 0.0178, 0.0185, 0.0181, 0.0183, 0.0198, 0.0186, 0.0232, 0.0229, 0.0178, 0.0176, 0.0175, 0.0198, 0.0204, 0.0166, 0.0206, 0.025, 0.0207, 0.0156, 0.0181, 0.0172, 0.0188, 0.0193, 0.0204, 0.0186, 0.018, 0.018, 0.0123, 0.0128, 0.015, 0.014, 0.0118, 0.0121, 0.0116, 0.0119, 0.0114, 0.0131, 0.0153, 0.0119, 0.0119, 0.0111, 0.0115, 0.0115, 0.0123, 0.0119, 0.0129, 0.0132, 0.0128, 0.0114, 0.0132, 0.0126, 0.0123, 0.0107, 0.0138, 0.0207, 0.0112, 0.0203, 0.0118, 0.0155, 0.0151, 0.0134, 0.015, 0.014, 0.017, 0.0157, 0.0158, 0.0269, 0.0163, 0.0152, 0.0176, 0.0162, 0.016, 0.0161, 0.0162, 0.0155, 0.0153, 0.0172, 0.0178, 0.0175, 0.0182, 0.0155, 0.0152, 0.0215, 0.0169, 0.0167, 0.0169, 0.0185, 0.0193, 0.0176, 0.0179, 0.0158, 0.0164, 0.019, 0.0189, 0.0206, 0.017, 0.0164, 0.0168, 0.0149, 0.0178, 0.0229, 0.0163, 0.0186, 0.0168, 0.0161, 0.016, 0.0189, 0.0164, 0.0167, 0.0223, 0.0179, 0.0156, 0.0207, 0.0183, 0.0178, 0.0154, 0.0158, 0.0184, 0.0202, 0.0157, 0.0169, 0.0165, 0.0168, 0.0173, 0.0195, 0.0172, 0.0209, 0.0211, 0.0211, 0.0174, 0.0213, 0.0203, 0.0188, 0.0175, 0.0157, 0.0181, 0.0192, 0.0196, 0.0269, 0.0161, 0.0153, 0.0166, 0.0171, 0.0151, 0.0158, 0.0166, 0.0168, 0.018, 0.0179, 0.018, 0.0188, 0.0178, 0.0161, 0.0174, 0.0159, 0.0189, 0.0185, 0.0158, 0.0196, 0.0213, 0.0161, 0.0164, 0.0162, 0.0157, 0.0176, 0.0176, 0.0179, 0.0194, 0.0204, 0.0171, 0.0165]

    #### 218MN
    Spatial_218MN = [0.0242, 0.0136, 0.012, 0.0251, 0.013, 0.0177, 0.0128, 0.0128, 0.018, 0.0159, 0.0149, 0.0146, 0.0152, 0.0295, 0.0182, 0.0157, 0.0152, 0.023, 0.0161, 0.0148, 0.015, 0.0138, 0.0143, 0.0225, 0.0146, 0.016, 0.0146, 0.0147, 0.015, 0.0154, 0.0149, 0.015, 0.0151, 0.0145, 0.015, 0.0219, 0.0143, 0.014, 0.0139, 0.0151, 0.0146, 0.015, 0.0152, 0.02, 0.0155, 0.0159, 0.0147, 0.017, 0.0162, 0.0157, 0.0158, 0.0158, 0.0284, 0.0179, 0.0163, 0.0165, 0.0149, 0.0161, 0.0158, 0.0156, 0.0166, 0.0166, 0.0155, 0.0141, 0.0159, 0.0158, 0.0161, 0.0161, 0.0161, 0.016, 0.0176, 0.0169, 0.028, 0.0158, 0.0168, 0.017, 0.016, 0.018, 0.016, 0.0157, 0.0162, 0.0148, 0.0177, 0.0266, 0.017, 0.0161, 0.0153, 0.0172, 0.0182, 0.0175, 0.0176, 0.0177, 0.0175, 0.0196, 0.019, 0.0172, 0.0175, 0.0196, 0.0184, 0.0268, 0.0165, 0.0154, 0.0149, 0.0167, 0.0161, 0.0159, 0.0161, 0.0154, 0.0186, 0.0171, 0.018, 0.018, 0.0192, 0.0189, 0.019, 0.0172, 0.0178, 0.0177, 0.019, 0.0202, 0.0137, 0.0131, 0.0142, 0.0133, 0.0133, 0.015, 0.0138, 0.0141, 0.0153, 0.0133, 0.0131, 0.0146, 0.013, 0.0131, 0.0131, 0.0133, 0.0134, 0.0134, 0.0138, 0.0135, 0.0298, 0.0141, 0.0129, 0.0135, 0.0146, 0.0127, 0.0151, 0.0176, 0.0171, 0.015, 0.0189, 0.0184, 0.0224, 0.0223, 0.0129, 0.0209, 0.0163, 0.0166, 0.0164, 0.0161, 0.0164, 0.0165, 0.0165, 0.017, 0.0171, 0.0178, 0.0196, 0.0205, 0.0167, 0.0179, 0.0195, 0.0165, 0.0166, 0.0193, 0.0161, 0.0167, 0.0186, 0.0175, 0.0171, 0.0164, 0.0181, 0.0203, 0.0187, 0.0193, 0.0239, 0.0288, 0.0171, 0.0174, 0.0172, 0.0231, 0.0213, 0.0159, 0.0197, 0.0189, 0.0156, 0.0179, 0.015, 0.015, 0.0155, 0.0148, 0.0184, 0.0202, 0.0182, 0.0155, 0.0161, 0.0167, 0.0162, 0.0159, 0.0142, 0.0168, 0.0146, 0.0152, 0.0178, 0.0168, 0.0187, 0.017, 0.0165, 0.0187, 0.0149, 0.0151, 0.0154, 0.0175, 0.0179, 0.0203, 0.0178, 0.0181, 0.0146, 0.0147, 0.0271, 0.0162, 0.017, 0.0176, 0.0199, 0.0193, 0.0175, 0.0199, 0.0197, 0.0191, 0.0169, 0.0197, 0.0171, 0.0175, 0.0172, 0.0153, 0.0187, 0.0149, 0.0146, 0.0143, 0.0192, 0.0181, 0.0142, 0.0153, 0.0145, 0.0167, 0.0146, 0.0139, 0.0183, 0.0154, 0.015, 0.0198, 0.0173]

    #### 21ML
    Spatial_21ML = [0.0117, 0.0219, 0.0142, 0.0145, 0.0179, 0.0258, 0.0158, 0.0276, 0.0116, 0.0174, 0.0136, 0.0131, 0.0138, 0.0266, 0.0167, 0.0122, 0.014, 0.0179, 0.0125, 0.0263, 0.0274, 0.0151, 0.0122, 0.0157, 0.0145, 0.0125, 0.0211, 0.012, 0.0156, 0.0163, 0.0158, 0.0118, 0.0253, 0.0149, 0.0208, 0.0278, 0.0222, 0.0127, 0.028, 0.0264, 0.0201, 0.0164, 0.0125, 0.0095, 0.0135, 0.0117, 0.0154, 0.0104, 0.0115, 0.0164, 0.0132, 0.0117]


    Spatial = Spatial_206FRL + Spatial_204FR + Spatial_203MN + Spatial_187FN + Spatial_211MRR + Spatial_63MR + Spatial_218MN + Spatial_54MRL + Spatial_21ML
    return Spatial

def data_sigma_1_5():
    #### 54MRL
    Spatial_54MRL = [0.0174, 0.0186, 0.018, 0.0306, 0.0266, 0.0267, 0.0168, 0.0236, 0.0251, 0.0355, 0.0319, 0.0237, 0.025, 0.0293, 0.0207, 0.0295, 0.0306, 0.017, 0.0275, 0.0219, 0.0207, 0.0184, 0.0231, 0.0206, 0.0166, 0.0205, 0.0395, 0.0339, 0.0235, 0.0197, 0.0239, 0.0384, 0.0227, 0.0263, 0.0186, 0.0231, 0.0436, 0.0262, 0.0216, 0.0265, 0.0196, 0.0264, 0.0254, 0.0348, 0.0219, 0.0196, 0.0189, 0.0224, 0.0187, 0.027, 0.0261, 0.0458, 0.0207, 0.0283, 0.0291, 0.0183, 0.0185, 0.0203, 0.028, 0.0257, 0.02, 0.0205, 0.0287, 0.0259, 0.0295, 0.0213, 0.0245, 0.0254, 0.0227, 0.0246, 0.0269, 0.0224, 0.0358, 0.033, 0.0267, 0.0323, 0.0301, 0.03, 0.0336, 0.0237, 0.0251, 0.0226, 0.0318, 0.0245, 0.027, 0.0243, 0.0402, 0.0266, 0.0216, 0.0255, 0.0274, 0.0244, 0.0186, 0.0309, 0.0252, 0.0367, 0.0336, 0.0264, 0.0223, 0.0233, 0.0347, 0.0245, 0.0257, 0.0187, 0.0337, 0.029, 0.0302, 0.0215, 0.0263, 0.0196, 0.0198, 0.0208, 0.0314, 0.0207, 0.0407, 0.0209, 0.0192, 0.02, 0.0267, 0.0219, 0.0198, 0.0195, 0.0182, 0.0347, 0.0217, 0.0219, 0.0251, 0.0185, 0.0305, 0.0199, 0.0311, 0.0208, 0.0198, 0.0206, 0.0328, 0.0364, 0.021, 0.0187, 0.0241, 0.0364, 0.0277, 0.021, 0.0215, 0.0324, 0.0309, 0.0188, 0.0265, 0.0247, 0.0265, 0.0332, 0.0204, 0.0204, 0.0235, 0.0207, 0.0309, 0.018, 0.0205, 0.0258, 0.0214, 0.0183, 0.022, 0.0331, 0.0313, 0.0169, 0.0207, 0.0212, 0.0205, 0.0316, 0.0381, 0.0234, 0.0273, 0.0426, 0.0221, 0.0222, 0.0194, 0.0299, 0.0248, 0.0379, 0.0229, 0.0243, 0.0184, 0.0188, 0.0373, 0.0211, 0.0196, 0.0192, 0.0191, 0.018, 0.0421, 0.0334, 0.0298, 0.0201, 0.0192, 0.0197, 0.0247, 0.0277, 0.0224, 0.0275, 0.0194, 0.0332, 0.0278, 0.0314, 0.043, 0.0285, 0.0176, 0.0201, 0.0254, 0.0177, 0.0432, 0.0258, 0.018, 0.0193, 0.0209, 0.0186, 0.0397, 0.0192, 0.0181, 0.0176, 0.0171, 0.0172, 0.0258, 0.0254, 0.0202, 0.0347, 0.0222, 0.0243, 0.0267, 0.0375, 0.0307, 0.0356, 0.0419, 0.0206, 0.0422, 0.0217, 0.0255, 0.0208, 0.0409, 0.0205, 0.0196, 0.0194, 0.0208, 0.0251, 0.0212, 0.024, 0.0237, 0.0241, 0.0229, 0.026, 0.0226, 0.0248, 0.0342, 0.0347, 0.0209, 0.0198, 0.0301, 0.0242, 0.0334, 0.023, 0.022, 0.0227, 0.0225, 0.0193, 0.0244, 0.0187, 0.0184, 0.0186, 0.0253, 0.0242, 0.0178, 0.0192, 0.0183, 0.0214, 0.0183, 0.0174, 0.024, 0.0196, 0.019, 0.0262, 0.0226]

    #### 63MR
    Spatial_63MR = [0.0223, 0.0153, 0.0152, 0.0162, 0.022, 0.0276, 0.0209, 0.0149, 0.0215, 0.0229, 0.0156, 0.0177, 0.0206, 0.0161, 0.0227, 0.0189, 0.0302, 0.0164, 0.016, 0.0304, 0.018, 0.018, 0.0172, 0.0234, 0.0159, 0.0217, 0.0162, 0.0157, 0.0258, 0.0272, 0.0154, 0.0183, 0.0205, 0.0216, 0.0236, 0.0204, 0.0321, 0.0193, 0.0235, 0.0288, 0.0238, 0.021, 0.0195, 0.0163, 0.0193, 0.0183, 0.0188, 0.02, 0.0247, 0.0203, 0.0179, 0.0175, 0.0212, 0.0286, 0.0282, 0.0225, 0.0242, 0.0179, 0.0326, 0.0195, 0.0256, 0.0223, 0.0202, 0.0305, 0.0393, 0.0187, 0.0235, 0.0184, 0.0194, 0.022, 0.0224, 0.0232, 0.0192, 0.0296, 0.0295, 0.017, 0.0323, 0.0212, 0.0195, 0.0223, 0.0293, 0.0219, 0.0206, 0.0176, 0.0156, 0.0142, 0.0156, 0.0268, 0.0167, 0.0147, 0.0169, 0.0386, 0.0305, 0.0195, 0.0244, 0.0155, 0.0305, 0.0197, 0.0242, 0.0393, 0.0231, 0.0275, 0.0231, 0.0314, 0.0235, 0.0261, 0.0271, 0.0322, 0.0234, 0.0239, 0.0258, 0.0161, 0.0177, 0.0205, 0.02, 0.0161, 0.0197, 0.0161, 0.0216, 0.0249, 0.027, 0.0225, 0.0177, 0.0276, 0.0393, 0.0472, 0.0148, 0.0172, 0.0164, 0.0148, 0.0168, 0.0198, 0.0259, 0.034, 0.0297, 0.0187, 0.0148, 0.0285, 0.0132, 0.0356, 0.0234, 0.0185, 0.0347, 0.0402, 0.0198, 0.0224, 0.0166, 0.0388, 0.0203, 0.0307, 0.0202, 0.0261, 0.0321, 0.0175, 0.0207, 0.0176, 0.0256, 0.0304, 0.0289, 0.0325, 0.0275, 0.0164, 0.018, 0.0169, 0.0191, 0.0214, 0.0182, 0.0392, 0.0213, 0.0172, 0.0324, 0.0173, 0.0204, 0.019, 0.0224, 0.0258, 0.0173, 0.0211, 0.05, 0.0177, 0.0184, 0.021, 0.0203, 0.0192, 0.024, 0.0178, 0.0368, 0.0255, 0.0203, 0.0218, 0.0178, 0.0178, 0.0187, 0.0184, 0.0199, 0.0253, 0.0347, 0.02, 0.0376, 0.0225, 0.0252, 0.012, 0.0324, 0.0287, 0.0295, 0.0392, 0.0178, 0.0135, 0.0281, 0.03, 0.023]

    #### 187FN
    Spatial_187FN = [0.0201, 0.0183, 0.0201, 0.0374, 0.0269, 0.0377, 0.0187, 0.0174, 0.0171, 0.0189, 0.0174, 0.0175, 0.0168, 0.0185, 0.0188, 0.0197, 0.0177, 0.0196, 0.0185, 0.0191, 0.0185, 0.0198, 0.0171, 0.0221, 0.0177, 0.0192, 0.0178, 0.0189, 0.0286, 0.0181, 0.0176, 0.0189, 0.0213, 0.0165, 0.0181, 0.0185, 0.0185, 0.0195, 0.0174, 0.0183, 0.0199, 0.0169, 0.0177, 0.0173, 0.0254, 0.0288, 0.0247, 0.0322, 0.0275, 0.0329, 0.0256, 0.0216, 0.0302, 0.026, 0.024, 0.032, 0.0237, 0.027, 0.0275, 0.0257, 0.0229, 0.0241, 0.0214, 0.0259, 0.0233, 0.0232, 0.0235, 0.0267, 0.023, 0.0211, 0.0288, 0.0284, 0.0235, 0.0269, 0.0261, 0.029, 0.0287, 0.0262, 0.0252, 0.0251, 0.0287, 0.0248, 0.0222, 0.0251, 0.0257, 0.0218, 0.0297, 0.0272, 0.0299, 0.043, 0.0262, 0.027, 0.025, 0.0231, 0.0346, 0.0284, 0.0286, 0.0292, 0.0344, 0.0275, 0.0332, 0.0189, 0.0221, 0.0185, 0.0196, 0.0227, 0.0247, 0.0174, 0.0169, 0.0177, 0.0191, 0.018, 0.0217, 0.0188, 0.0187, 0.0206, 0.0197, 0.0289, 0.0159, 0.0182, 0.0216, 0.0235, 0.018, 0.0234, 0.0197, 0.0254, 0.0243, 0.0189, 0.0245, 0.0483, 0.0191, 0.018, 0.0184, 0.0166, 0.0151, 0.0217, 0.0204, 0.0246, 0.0197, 0.0278, 0.0234, 0.0248, 0.0179, 0.0211, 0.0255, 0.022, 0.0237, 0.026, 0.0258, 0.0223, 0.0185, 0.019, 0.0189, 0.0249, 0.0228, 0.0219, 0.0263, 0.0255, 0.0246, 0.0207, 0.0222, 0.0186, 0.0242, 0.0209, 0.0252, 0.024, 0.0261, 0.0276, 0.0223, 0.029, 0.0283, 0.0202, 0.0355, 0.0193, 0.0239, 0.0223, 0.0221, 0.0216, 0.0226, 0.0311, 0.0318, 0.0272, 0.0249, 0.0292, 0.0294, 0.0226, 0.0316, 0.0221, 0.0283, 0.0302, 0.0191, 0.021, 0.0243, 0.0181, 0.034, 0.0233, 0.0334, 0.0229, 0.0205, 0.021, 0.0252, 0.0209, 0.0198, 0.0238, 0.0459, 0.023, 0.0219, 0.0241, 0.0302, 0.0218, 0.0219, 0.0262, 0.0177, 0.022, 0.0225, 0.0235, 0.0197, 0.0206, 0.021, 0.0202, 0.0222, 0.0232, 0.0196, 0.0195, 0.0292, 0.0229, 0.0257, 0.0237, 0.0192, 0.0229, 0.0237, 0.0245, 0.0277, 0.0263, 0.0286, 0.0194, 0.0224, 0.0209, 0.0261, 0.0202, 0.0195, 0.0195, 0.0217, 0.0188, 0.0186, 0.0216, 0.0274, 0.0246, 0.0281, 0.0246, 0.0232, 0.0186, 0.0195, 0.0455, 0.0203, 0.019, 0.0211, 0.023, 0.0225, 0.0195, 0.0199, 0.0168, 0.0217, 0.0261, 0.0178, 0.0273, 0.0229, 0.022, 0.0262, 0.0252, 0.0249, 0.0366, 0.0317, 0.0245, 0.0233, 0.0246, 0.0212, 0.0222, 0.0276, 0.021, 0.0196, 0.0245, 0.0215, 0.0243, 0.026, 0.026, 0.0258, 0.024, 0.022, 0.0223, 0.024, 0.0242, 0.0272, 0.0419, 0.0208, 0.0226, 0.0199, 0.0308, 0.0228, 0.0213, 0.026, 0.0239, 0.0273, 0.0251, 0.0314, 0.0274, 0.0268, 0.0234, 0.0305, 0.0278, 0.0246, 0.0353, 0.0282, 0.028, 0.0207, 0.0236, 0.0347, 0.0442, 0.0261, 0.0205, 0.0168, 0.0296, 0.0186, 0.0176, 0.0189, 0.0195, 0.0186, 0.0173, 0.023, 0.0186, 0.0254, 0.0164, 0.0193, 0.0187, 0.0204, 0.0336, 0.0188, 0.0207, 0.0347, 0.024, 0.0171, 0.0223, 0.0211, 0.0213, 0.0227, 0.025, 0.0297, 0.0333, 0.0221, 0.0359, 0.024, 0.0434, 0.0167, 0.02, 0.0179, 0.0192, 0.0184, 0.0177, 0.019, 0.0173, 0.02, 0.0419, 0.0374, 0.0272, 0.0317, 0.0182, 0.0157, 0.0235, 0.0197, 0.0165, 0.0194, 0.0168, 0.0322, 0.0174, 0.0173, 0.046, 0.0218, 0.0179, 0.0204, 0.0162, 0.0178, 0.0182, 0.0217, 0.0182, 0.0189]

    #### 203MN
    Spatial_203MN = [0.0194, 0.0205, 0.0221, 0.0181, 0.0189, 0.0194, 0.0197, 0.0193, 0.0225, 0.0203, 0.0188, 0.022, 0.0228, 0.0195, 0.0329, 0.0233, 0.0389, 0.0202, 0.0193, 0.0199, 0.0218, 0.0221, 0.0197, 0.0189, 0.0241, 0.0207, 0.0204, 0.02, 0.0206, 0.0198, 0.0202, 0.0207, 0.02, 0.0204, 0.03, 0.0253, 0.0257, 0.0262, 0.0281, 0.0247, 0.0267, 0.0275, 0.0257, 0.0262, 0.0261, 0.0282, 0.026, 0.0264, 0.0279, 0.0277, 0.0264, 0.0251, 0.0284, 0.0271, 0.027, 0.0251, 0.0246, 0.025, 0.0279, 0.0269, 0.0265, 0.0265, 0.0256, 0.0423, 0.029, 0.0271, 0.0245, 0.0252, 0.0257, 0.0269, 0.027, 0.0266, 0.0266, 0.0265, 0.0272, 0.0281, 0.0267, 0.0254, 0.0304, 0.025, 0.0267, 0.0265, 0.0249, 0.0348, 0.0247, 0.0269, 0.0248, 0.0253, 0.0262, 0.0278, 0.0188, 0.0185, 0.0192, 0.0195, 0.0199, 0.0252, 0.02, 0.0203, 0.0192, 0.0196, 0.0199, 0.0196, 0.0238, 0.0427, 0.0184, 0.0183, 0.0231, 0.0175, 0.0218, 0.0204, 0.0196, 0.0196, 0.0186, 0.0206, 0.0185, 0.0185, 0.0209, 0.0234, 0.0197, 0.0194, 0.02, 0.0235, 0.0202, 0.02, 0.0186, 0.0185, 0.0408, 0.0317, 0.0229, 0.0186, 0.0188, 0.02, 0.0204, 0.0222, 0.0236, 0.0241, 0.0239, 0.0235, 0.0238, 0.0244, 0.0227, 0.024, 0.0234, 0.0212, 0.0241, 0.0261, 0.0242, 0.0243, 0.0256, 0.0243, 0.0224, 0.0245, 0.0229, 0.0239, 0.0222, 0.0281, 0.0236, 0.0231, 0.0267, 0.0262, 0.0261, 0.0231, 0.0252, 0.0218, 0.0455, 0.0498, 0.0246, 0.0211, 0.0231, 0.0232, 0.022, 0.024, 0.0235, 0.0214, 0.0249, 0.0229, 0.0275, 0.0225, 0.0239, 0.0218, 0.0243, 0.0252, 0.0208, 0.022, 0.0244, 0.031, 0.025, 0.025, 0.0387, 0.0208, 0.0267, 0.0234, 0.0202, 0.021, 0.0208, 0.0284, 0.0237, 0.0212, 0.0217, 0.0222, 0.0212, 0.0222, 0.0219, 0.0217, 0.0231, 0.0221, 0.0217, 0.0268, 0.0199, 0.0211, 0.0215, 0.0216, 0.0225, 0.0216, 0.0235, 0.0221, 0.0255, 0.0252, 0.0236, 0.0217, 0.024, 0.0244, 0.0346, 0.0215, 0.0259, 0.0203, 0.0208, 0.0216, 0.0222, 0.021, 0.0191, 0.0189, 0.0221, 0.0207, 0.0222, 0.0193, 0.0244, 0.0235, 0.0196, 0.0192, 0.0183, 0.0186, 0.0206, 0.0196, 0.0209, 0.0332, 0.0203, 0.0187, 0.0192, 0.0196, 0.0191, 0.0181, 0.019, 0.0185, 0.0202, 0.0269, 0.0197, 0.0193, 0.0455, 0.0286, 0.0362, 0.0243, 0.0211, 0.0174, 0.0186, 0.0204, 0.0221, 0.0194, 0.0219, 0.0195, 0.0231, 0.0198, 0.0192, 0.0354, 0.0284, 0.0207, 0.02, 0.0223, 0.0197, 0.0209, 0.0238, 0.0235, 0.0238, 0.0223, 0.0233, 0.0241, 0.02, 0.0198, 0.0223, 0.0179, 0.0249, 0.0216, 0.0253, 0.0375, 0.0208, 0.0209, 0.0195, 0.0197, 0.0287, 0.0211, 0.0277, 0.0296, 0.0462, 0.0264, 0.0195, 0.0178, 0.021, 0.0187, 0.0237, 0.022, 0.0216, 0.0225, 0.0189, 0.0214, 0.0245, 0.0212, 0.0247, 0.0196, 0.0191, 0.0175, 0.0173, 0.0418, 0.0178, 0.0193, 0.0199, 0.0208, 0.0206, 0.0193, 0.0204, 0.021, 0.0219, 0.0316]

    #### 204FR
    Spatial_204FR = [0.0168, 0.039, 0.0178, 0.0311, 0.036, 0.026, 0.0158, 0.0164, 0.0165, 0.0174, 0.0184, 0.0162, 0.0339, 0.0201, 0.0206, 0.0208, 0.0195, 0.0386, 0.0292, 0.0321, 0.0364, 0.0244, 0.0222, 0.0188, 0.0181, 0.0183, 0.0206, 0.0215, 0.0175, 0.0197, 0.0406, 0.032, 0.0407, 0.0318, 0.02, 0.0191, 0.0191, 0.0198, 0.0203, 0.0202, 0.019, 0.0271, 0.0197, 0.0185, 0.0191, 0.0239, 0.018, 0.0228, 0.0218, 0.0233, 0.0233, 0.0222, 0.0244, 0.0237, 0.026, 0.0201, 0.0221, 0.0211, 0.0212, 0.0426, 0.0238, 0.023, 0.0227, 0.0245, 0.0242, 0.0233, 0.0216, 0.0203, 0.0253, 0.0241, 0.0233, 0.0239, 0.0245, 0.0294, 0.0237, 0.024, 0.0471, 0.0398, 0.0333, 0.0217, 0.0217, 0.0233, 0.0229, 0.0232, 0.0321, 0.0226, 0.0222, 0.0223, 0.0244, 0.0244, 0.0239, 0.0263, 0.0228, 0.0224, 0.0215, 0.0385, 0.0266, 0.02, 0.0212, 0.0234, 0.0317, 0.0306, 0.0328, 0.0315, 0.031, 0.0327, 0.0311, 0.0313, 0.0336, 0.0309, 0.0303, 0.0327, 0.032, 0.0313, 0.0317, 0.0307, 0.0318, 0.0309, 0.0292, 0.0293, 0.0323, 0.0309, 0.0326, 0.0313, 0.03, 0.0319, 0.0316, 0.0317, 0.0294, 0.0324, 0.0248, 0.0233, 0.025, 0.0246, 0.0245, 0.0229, 0.0234, 0.0233, 0.0213, 0.0227, 0.0217, 0.0214, 0.021, 0.0258, 0.0282, 0.0314, 0.0253, 0.0234, 0.0242, 0.0233, 0.0241, 0.0221, 0.0235, 0.024, 0.0502, 0.0419, 0.0219, 0.0395, 0.0233, 0.0226, 0.0224, 0.0226, 0.0252, 0.0242, 0.0246, 0.0224, 0.0232, 0.0239, 0.0235, 0.0244, 0.0234, 0.0254, 0.0272, 0.0226, 0.0225, 0.022, 0.0257, 0.0236, 0.0339, 0.0377, 0.0284, 0.0244, 0.0238, 0.0232, 0.0264, 0.0253, 0.0245, 0.026, 0.0253, 0.0247, 0.0247, 0.0252, 0.0245, 0.0251, 0.0238, 0.0262, 0.0249, 0.0228, 0.0236, 0.0221, 0.0242, 0.0222, 0.0277, 0.0242, 0.0236, 0.021, 0.0242, 0.0248, 0.0375, 0.0273, 0.028, 0.0273, 0.0248, 0.0222, 0.0293, 0.0235, 0.0277, 0.0326, 0.0218, 0.0234, 0.0238, 0.0244, 0.0244, 0.0266, 0.0267, 0.0249, 0.0242, 0.0233, 0.0247, 0.0252, 0.0261, 0.0254, 0.025, 0.0256, 0.024, 0.0256, 0.0272, 0.0288, 0.0236, 0.0254, 0.0278, 0.0239, 0.0231, 0.0242, 0.0256, 0.0243, 0.0233, 0.0238, 0.0298, 0.0304, 0.0496, 0.041, 0.0238, 0.0224, 0.0226, 0.0229, 0.0236, 0.0269, 0.0249, 0.0276, 0.0257, 0.0319, 0.0344, 0.0232, 0.0303, 0.0252, 0.0246, 0.0243, 0.0245, 0.0249, 0.028, 0.0207, 0.0296, 0.0298, 0.0288, 0.0223, 0.0201, 0.0196, 0.0197, 0.0225, 0.0343, 0.0348, 0.0277, 0.0267, 0.0223, 0.0223, 0.0226, 0.0225]

    #### 206FRL
    Spatial_206FRL = [0.0208, 0.0205, 0.0234, 0.0217, 0.0235, 0.0191, 0.0213, 0.0253, 0.0197, 0.0205, 0.0201, 0.0202, 0.0198, 0.0205, 0.0237, 0.0197, 0.0201, 0.0203, 0.0189, 0.0203, 0.0204, 0.0208, 0.0226, 0.0205, 0.0189, 0.0204, 0.0204, 0.0198, 0.0194, 0.0306, 0.0262, 0.0191, 0.0201, 0.0226, 0.0224, 0.02, 0.0195, 0.0194, 0.0186, 0.0227, 0.0196, 0.0178, 0.0225, 0.0206, 0.0181, 0.0187, 0.018, 0.0195, 0.0164, 0.0176, 0.0188, 0.0181, 0.0177, 0.0261, 0.0273, 0.0188, 0.034, 0.0275, 0.0169, 0.0181, 0.0208, 0.0207, 0.0341, 0.0226, 0.0194, 0.0184, 0.0189, 0.0203, 0.0189, 0.0186, 0.0194, 0.0202, 0.0185, 0.019, 0.0192, 0.0254, 0.0192, 0.0214, 0.018, 0.0199, 0.0185, 0.0191, 0.0189, 0.0205, 0.0235, 0.0178, 0.0188, 0.0367, 0.032, 0.0332, 0.0295, 0.0316, 0.0181, 0.0202, 0.0188, 0.0202, 0.0201, 0.0175, 0.0175, 0.0209, 0.0202, 0.0182, 0.0216, 0.0187, 0.0208, 0.0292, 0.0204, 0.02, 0.0186, 0.0196, 0.0194, 0.0194, 0.0172, 0.0203, 0.0352, 0.029, 0.0241, 0.0248, 0.0238, 0.021, 0.0255, 0.0251, 0.0244, 0.0244, 0.0245, 0.0253, 0.0237, 0.0282, 0.0266, 0.0263, 0.0239, 0.0263, 0.0237, 0.0244, 0.026, 0.0254, 0.0249, 0.033, 0.0233, 0.0242, 0.0238, 0.0241, 0.0238, 0.0245, 0.0249, 0.0251, 0.026, 0.0241, 0.0244, 0.0241, 0.0244, 0.0238, 0.024, 0.0239, 0.0235, 0.0249, 0.0219, 0.0243, 0.0228, 0.0236, 0.0216, 0.023, 0.0217, 0.0214, 0.0216, 0.0234, 0.0212, 0.0219, 0.0234, 0.0273, 0.02, 0.021, 0.0211, 0.022, 0.0223, 0.021, 0.0217, 0.022, 0.0219, 0.0221, 0.0215, 0.0217, 0.0223, 0.0218, 0.0223, 0.0215, 0.0231, 0.0209, 0.0216, 0.0217, 0.0305, 0.026, 0.0255, 0.0205, 0.0213, 0.0218, 0.0221, 0.0411, 0.0272, 0.0277, 0.0237, 0.0214, 0.0211, 0.0217, 0.0216, 0.0224, 0.0235, 0.0215, 0.0194, 0.0229, 0.0287, 0.0214, 0.0215, 0.0242, 0.024, 0.0231, 0.0229, 0.0233, 0.028, 0.0223, 0.0232, 0.0235, 0.0238, 0.0234, 0.024, 0.0246, 0.0241, 0.024, 0.0232, 0.0232, 0.0256, 0.0254, 0.0255, 0.0273, 0.0297, 0.0213, 0.0216, 0.0233, 0.0246, 0.0223, 0.0229, 0.0248, 0.0222, 0.0231, 0.0232, 0.0219, 0.023, 0.0226, 0.0221, 0.022, 0.0227, 0.0213, 0.0222, 0.0222, 0.0403, 0.0353, 0.0278, 0.0235, 0.0221, 0.0236, 0.022, 0.0196, 0.026, 0.0247, 0.0232, 0.0235, 0.0224, 0.0333, 0.0214, 0.0214, 0.0232, 0.0253, 0.0246, 0.0244, 0.0245, 0.0233, 0.0239, 0.0248, 0.0225, 0.0224, 0.0223, 0.0243, 0.0229, 0.0254, 0.0233, 0.0206, 0.0222, 0.0222, 0.0244, 0.0231, 0.024, 0.0222, 0.0219, 0.0314, 0.0335, 0.0216, 0.0213, 0.0223, 0.0217, 0.0222, 0.0211, 0.0209, 0.0222, 0.0217, 0.0257, 0.0279, 0.0249, 0.0247, 0.0231, 0.0228, 0.0222, 0.0239, 0.0241, 0.0229, 0.0232, 0.0226, 0.0225, 0.0298, 0.03, 0.0318, 0.0317, 0.0326, 0.0321, 0.0365, 0.0354, 0.0328, 0.0324, 0.0343, 0.0339, 0.0371, 0.0326]

    #### 211MRR
    Spatial_211MRR = [0.023, 0.023, 0.0238, 0.0224, 0.0227, 0.0219, 0.024, 0.024, 0.0225, 0.0254, 0.0238, 0.0236, 0.0238, 0.0303, 0.0221, 0.0227, 0.0245, 0.0247, 0.0233, 0.0249, 0.0271, 0.0245, 0.0259, 0.0292, 0.0282, 0.0243, 0.0283, 0.0257, 0.0271, 0.0236, 0.0206, 0.0243, 0.023, 0.0225, 0.0218, 0.0225, 0.0232, 0.0243, 0.0255, 0.0266, 0.0287, 0.0245, 0.0253, 0.0228, 0.0211, 0.0204, 0.0201, 0.0229, 0.024, 0.0252, 0.0208, 0.0233, 0.0206, 0.0242, 0.0182, 0.0239, 0.0207, 0.02, 0.0206, 0.0194, 0.0222, 0.0239, 0.0289, 0.0214, 0.0245, 0.0249, 0.0238, 0.0262, 0.0213, 0.0215, 0.0219, 0.022, 0.0235, 0.0212, 0.0196, 0.0251, 0.0247, 0.0221, 0.0248, 0.0228, 0.0256, 0.0258, 0.0251, 0.0193, 0.0261, 0.0231, 0.0165, 0.0157, 0.017, 0.0195, 0.0206, 0.0245, 0.0165, 0.0158, 0.0172, 0.0196, 0.017, 0.017, 0.0165, 0.0184, 0.0185, 0.017, 0.0189, 0.0161, 0.0177, 0.0185, 0.019, 0.0183, 0.0161, 0.0168, 0.0158, 0.017, 0.0178, 0.0165, 0.0229, 0.0349, 0.0253, 0.0265, 0.0281, 0.0274, 0.0202, 0.0261, 0.0247, 0.0249, 0.0207, 0.0286, 0.0215, 0.0227, 0.0223, 0.0224, 0.0248, 0.0229, 0.0305, 0.0299, 0.0213, 0.0211, 0.0213, 0.0249, 0.0259, 0.0199, 0.026, 0.0333, 0.0261, 0.0183, 0.0222, 0.0206, 0.0232, 0.0241, 0.0259, 0.0229, 0.0218, 0.0217, 0.0154, 0.0164, 0.0201, 0.0183, 0.0146, 0.0154, 0.0144, 0.015, 0.0142, 0.0171, 0.0204, 0.0151, 0.0147, 0.0134, 0.0142, 0.0143, 0.0155, 0.015, 0.0166, 0.0169, 0.0165, 0.0142, 0.0171, 0.0162, 0.0157, 0.013, 0.0181, 0.0288, 0.0138, 0.0278, 0.0147, 0.0207, 0.0199, 0.0173, 0.0196, 0.0181, 0.0209, 0.0189, 0.0193, 0.0367, 0.0199, 0.0181, 0.0219, 0.0197, 0.0198, 0.0194, 0.0199, 0.0188, 0.0187, 0.0216, 0.0225, 0.0221, 0.023, 0.0188, 0.0181, 0.0284, 0.0205, 0.0208, 0.0214, 0.0236, 0.025, 0.0223, 0.0227, 0.0195, 0.0202, 0.0244, 0.0243, 0.027, 0.0211, 0.0201, 0.0208, 0.018, 0.0224, 0.0309, 0.0206, 0.0238, 0.0207, 0.0198, 0.0194, 0.0243, 0.0204, 0.0207, 0.0297, 0.023, 0.0194, 0.0275, 0.0237, 0.0223, 0.0185, 0.0195, 0.0237, 0.0265, 0.0188, 0.0211, 0.0206, 0.0211, 0.0217, 0.0254, 0.0217, 0.0278, 0.0281, 0.0279, 0.0218, 0.0282, 0.0266, 0.0243, 0.0223, 0.0201, 0.0232, 0.025, 0.0253, 0.0367, 0.0201, 0.0185, 0.0207, 0.0215, 0.0186, 0.0193, 0.0205, 0.021, 0.0233, 0.023, 0.0232, 0.0244, 0.0228, 0.02, 0.0219, 0.0197, 0.0246, 0.0239, 0.0194, 0.0254, 0.0278, 0.0195, 0.0201, 0.02, 0.0189, 0.0223, 0.0222, 0.0224, 0.0252, 0.0267, 0.0216, 0.0205]

    #### 218MN
    Spatial_218MN = [0.0337, 0.0169, 0.0146, 0.0346, 0.0159, 0.0238, 0.0157, 0.0157, 0.0241, 0.0199, 0.0182, 0.0175, 0.0188, 0.0411, 0.0235, 0.0195, 0.0183, 0.0312, 0.0199, 0.0179, 0.0182, 0.0164, 0.0171, 0.0304, 0.018, 0.0198, 0.0175, 0.0178, 0.0182, 0.0189, 0.018, 0.0182, 0.0184, 0.0175, 0.0185, 0.0295, 0.0172, 0.017, 0.0166, 0.0184, 0.0178, 0.0181, 0.0182, 0.026, 0.0189, 0.0193, 0.0177, 0.021, 0.0198, 0.0191, 0.019, 0.0194, 0.0391, 0.0229, 0.0198, 0.02, 0.0177, 0.0194, 0.019, 0.0188, 0.0204, 0.0204, 0.0187, 0.0168, 0.0193, 0.0192, 0.0197, 0.0197, 0.0197, 0.0193, 0.0221, 0.0211, 0.0384, 0.0191, 0.0208, 0.0212, 0.0196, 0.0229, 0.0196, 0.0192, 0.0196, 0.0176, 0.0223, 0.0363, 0.021, 0.0194, 0.0183, 0.0217, 0.0231, 0.022, 0.0221, 0.0224, 0.0221, 0.0257, 0.0245, 0.0215, 0.022, 0.0257, 0.0233, 0.0365, 0.0202, 0.0187, 0.018, 0.0205, 0.0196, 0.0193, 0.0197, 0.0188, 0.0237, 0.0215, 0.0229, 0.0229, 0.0249, 0.0244, 0.0245, 0.0215, 0.0226, 0.0226, 0.0248, 0.0264, 0.017, 0.016, 0.0175, 0.0162, 0.0163, 0.0193, 0.0174, 0.0177, 0.0195, 0.0162, 0.016, 0.0187, 0.0159, 0.0159, 0.0158, 0.0162, 0.0165, 0.0164, 0.0172, 0.0165, 0.0418, 0.0175, 0.0159, 0.0166, 0.0185, 0.0157, 0.0193, 0.0234, 0.0225, 0.019, 0.0254, 0.0247, 0.0306, 0.0306, 0.0156, 0.0284, 0.0196, 0.0201, 0.0199, 0.0194, 0.0198, 0.02, 0.0204, 0.0209, 0.0211, 0.0223, 0.0256, 0.0268, 0.0207, 0.0225, 0.0251, 0.0199, 0.0202, 0.0246, 0.0195, 0.0205, 0.0237, 0.0218, 0.021, 0.0197, 0.023, 0.0266, 0.0238, 0.0249, 0.032, 0.0393, 0.0213, 0.0217, 0.022, 0.0309, 0.028, 0.0193, 0.0257, 0.0244, 0.0192, 0.0227, 0.0184, 0.0184, 0.0196, 0.0184, 0.0241, 0.0269, 0.024, 0.0197, 0.0202, 0.0213, 0.0204, 0.02, 0.0172, 0.0214, 0.0178, 0.019, 0.0232, 0.0217, 0.0248, 0.0221, 0.0216, 0.0247, 0.0183, 0.0185, 0.0194, 0.0228, 0.0237, 0.027, 0.0232, 0.0234, 0.018, 0.018, 0.0375, 0.0207, 0.022, 0.0229, 0.0266, 0.0257, 0.0228, 0.0265, 0.0261, 0.0252, 0.022, 0.0263, 0.0224, 0.0229, 0.0226, 0.0194, 0.0247, 0.0185, 0.0182, 0.0181, 0.0257, 0.0241, 0.0177, 0.0193, 0.0182, 0.0216, 0.0186, 0.0172, 0.0242, 0.0195, 0.0189, 0.0263, 0.0224]

    #### 21ML
    Spatial_21ML = [0.0149, 0.0299, 0.0183, 0.0193, 0.0241, 0.0354, 0.0206, 0.0389, 0.0149, 0.0237, 0.0176, 0.0172, 0.0175, 0.037, 0.0219, 0.0149, 0.0178, 0.0239, 0.0154, 0.037, 0.0381, 0.0198, 0.0149, 0.0206, 0.0185, 0.0153, 0.0287, 0.0146, 0.0201, 0.0215, 0.0207, 0.0142, 0.0352, 0.0203, 0.029, 0.039, 0.0302, 0.0169, 0.0396, 0.0376, 0.0281, 0.0224, 0.0166, 0.0118, 0.0179, 0.0152, 0.0213, 0.0133, 0.0152, 0.0221, 0.0176, 0.0152]

    Spatial = Spatial_206FRL + Spatial_204FR + Spatial_203MN + Spatial_187FN + Spatial_211MRR + Spatial_63MR + Spatial_218MN + Spatial_54MRL + Spatial_21ML
    return Spatial

def data_sigma_2():
    #### 54MRL
    Spatial_54MRL = [0.0201, 0.0215, 0.0208, 0.0376, 0.0323, 0.0324, 0.0191, 0.0282, 0.0303, 0.0434, 0.039, 0.0281, 0.0299, 0.0355, 0.0244, 0.0356, 0.0368, 0.0195, 0.0329, 0.0265, 0.0246, 0.0213, 0.0275, 0.0246, 0.0188, 0.0239, 0.0487, 0.0411, 0.0277, 0.0229, 0.0283, 0.0462, 0.0267, 0.0312, 0.0212, 0.0271, 0.0513, 0.0307, 0.0252, 0.0318, 0.0226, 0.0311, 0.0302, 0.0426, 0.0258, 0.0224, 0.0214, 0.0263, 0.0214, 0.0325, 0.0308, 0.0566, 0.024, 0.0338, 0.0351, 0.021, 0.0212, 0.0236, 0.0334, 0.0307, 0.0232, 0.0239, 0.0341, 0.0309, 0.0355, 0.025, 0.0289, 0.0295, 0.0263, 0.0286, 0.0314, 0.0255, 0.0425, 0.0397, 0.0314, 0.0387, 0.0351, 0.0359, 0.0402, 0.0274, 0.029, 0.0258, 0.0376, 0.028, 0.0322, 0.0281, 0.0475, 0.0305, 0.0244, 0.0297, 0.0322, 0.0285, 0.0206, 0.0364, 0.0294, 0.0444, 0.0405, 0.0307, 0.0254, 0.0265, 0.0417, 0.0276, 0.03, 0.0208, 0.0402, 0.0326, 0.0349, 0.0247, 0.031, 0.0221, 0.0223, 0.0238, 0.0375, 0.0236, 0.0496, 0.024, 0.0217, 0.0225, 0.0314, 0.025, 0.0226, 0.0219, 0.0203, 0.0417, 0.025, 0.0251, 0.0293, 0.0207, 0.0363, 0.0224, 0.0372, 0.0238, 0.0223, 0.0235, 0.0392, 0.0434, 0.024, 0.0209, 0.0276, 0.0437, 0.0331, 0.024, 0.0246, 0.0388, 0.0367, 0.0214, 0.0313, 0.0289, 0.0316, 0.0401, 0.0233, 0.0232, 0.0276, 0.0237, 0.0368, 0.02, 0.0234, 0.0305, 0.0248, 0.0204, 0.0252, 0.0392, 0.0375, 0.0186, 0.0235, 0.0245, 0.0233, 0.0382, 0.0466, 0.0271, 0.0322, 0.0519, 0.0256, 0.0258, 0.022, 0.0353, 0.0291, 0.0455, 0.0269, 0.0288, 0.0206, 0.021, 0.0452, 0.0239, 0.0223, 0.0217, 0.0221, 0.0206, 0.0518, 0.0402, 0.0355, 0.0234, 0.0222, 0.0227, 0.0295, 0.033, 0.0266, 0.0333, 0.0224, 0.0405, 0.0333, 0.0375, 0.0515, 0.0339, 0.0199, 0.0232, 0.0306, 0.0201, 0.0508, 0.03, 0.0203, 0.0223, 0.0245, 0.0214, 0.049, 0.022, 0.0206, 0.0199, 0.0193, 0.0195, 0.0311, 0.0304, 0.023, 0.0417, 0.0256, 0.0286, 0.0314, 0.0454, 0.0366, 0.043, 0.051, 0.0234, 0.0508, 0.0248, 0.0292, 0.0237, 0.0495, 0.0231, 0.0222, 0.0219, 0.0237, 0.0295, 0.0244, 0.0279, 0.0275, 0.0281, 0.0266, 0.0306, 0.0263, 0.029, 0.0413, 0.0416, 0.0239, 0.0223, 0.036, 0.0282, 0.0397, 0.0268, 0.0265, 0.0273, 0.0272, 0.0226, 0.0294, 0.0216, 0.0213, 0.0219, 0.0309, 0.0293, 0.0206, 0.0224, 0.0214, 0.0254, 0.0215, 0.02, 0.0288, 0.023, 0.0222, 0.0316, 0.0268]

    #### 63MR
    Spatial_63MR = [0.0267, 0.0173, 0.0172, 0.0183, 0.0261, 0.0337, 0.0246, 0.0166, 0.0254, 0.0266, 0.0177, 0.0205, 0.0245, 0.0185, 0.0274, 0.0222, 0.0365, 0.019, 0.0184, 0.0374, 0.0208, 0.021, 0.0197, 0.0281, 0.0182, 0.026, 0.0185, 0.0178, 0.0313, 0.0327, 0.0174, 0.0207, 0.0236, 0.025, 0.0281, 0.0233, 0.0391, 0.0221, 0.0277, 0.0346, 0.0277, 0.0238, 0.0224, 0.0179, 0.0218, 0.0204, 0.0211, 0.0229, 0.0293, 0.0233, 0.02, 0.0196, 0.0245, 0.0347, 0.0337, 0.0262, 0.0286, 0.02, 0.0396, 0.0224, 0.0306, 0.026, 0.0231, 0.0371, 0.0486, 0.0218, 0.0278, 0.0207, 0.022, 0.0257, 0.0263, 0.0273, 0.0231, 0.0363, 0.0361, 0.0198, 0.0391, 0.0254, 0.0232, 0.0269, 0.0354, 0.0265, 0.0247, 0.0205, 0.018, 0.0161, 0.0179, 0.0328, 0.0193, 0.0168, 0.0196, 0.0479, 0.0376, 0.0232, 0.0298, 0.0178, 0.0375, 0.0232, 0.0293, 0.0488, 0.0276, 0.0336, 0.0275, 0.0376, 0.0281, 0.0317, 0.033, 0.0392, 0.0277, 0.0287, 0.0314, 0.0182, 0.0204, 0.024, 0.0235, 0.0181, 0.0231, 0.0182, 0.0257, 0.0301, 0.0328, 0.0267, 0.0203, 0.0337, 0.0487, 0.0588, 0.0166, 0.0203, 0.0192, 0.0171, 0.0198, 0.0239, 0.0319, 0.0424, 0.0367, 0.0223, 0.0171, 0.0351, 0.0151, 0.0441, 0.0286, 0.0223, 0.043, 0.0501, 0.024, 0.0272, 0.0187, 0.0478, 0.0239, 0.0376, 0.0235, 0.0311, 0.0388, 0.02, 0.0242, 0.0201, 0.0306, 0.037, 0.0348, 0.0386, 0.0331, 0.0185, 0.0206, 0.019, 0.022, 0.0253, 0.0209, 0.0481, 0.025, 0.0195, 0.0397, 0.0196, 0.0239, 0.0217, 0.0266, 0.0311, 0.0198, 0.0243, 0.0613, 0.0199, 0.0209, 0.0244, 0.0233, 0.0221, 0.0286, 0.0201, 0.0451, 0.0304, 0.0237, 0.0255, 0.0201, 0.0202, 0.0212, 0.021, 0.023, 0.0303, 0.0422, 0.0229, 0.0459, 0.0264, 0.031, 0.0135, 0.0398, 0.035, 0.0359, 0.049, 0.0211, 0.0154, 0.0346, 0.0371, 0.028]

    #### 187FN
    Spatial_187FN = [0.0235, 0.0209, 0.0234, 0.0466, 0.0329, 0.0468, 0.0217, 0.0197, 0.0193, 0.0217, 0.0197, 0.0199, 0.0191, 0.0213, 0.0216, 0.0228, 0.0201, 0.0231, 0.0214, 0.022, 0.0213, 0.023, 0.0197, 0.0261, 0.0203, 0.0223, 0.0203, 0.0217, 0.0351, 0.0206, 0.02, 0.0217, 0.0252, 0.0188, 0.0207, 0.0213, 0.0211, 0.0227, 0.0197, 0.0209, 0.0232, 0.0191, 0.0201, 0.0196, 0.0293, 0.0335, 0.0285, 0.0382, 0.0321, 0.0392, 0.0294, 0.0242, 0.0359, 0.0301, 0.0272, 0.0383, 0.0268, 0.0314, 0.0321, 0.0297, 0.0257, 0.0275, 0.0237, 0.03, 0.0264, 0.0263, 0.0266, 0.0309, 0.0261, 0.0238, 0.0335, 0.0333, 0.0268, 0.0312, 0.0302, 0.0342, 0.0336, 0.0304, 0.029, 0.0288, 0.0335, 0.0284, 0.0249, 0.0288, 0.0296, 0.0245, 0.035, 0.0316, 0.0353, 0.0528, 0.0303, 0.0313, 0.0286, 0.026, 0.0416, 0.0334, 0.0335, 0.0346, 0.0411, 0.032, 0.0398, 0.0222, 0.0262, 0.0215, 0.0229, 0.0273, 0.0296, 0.02, 0.0191, 0.0204, 0.0221, 0.0207, 0.0257, 0.0218, 0.0218, 0.0241, 0.0232, 0.0351, 0.018, 0.0212, 0.0255, 0.028, 0.0207, 0.0279, 0.0232, 0.0306, 0.029, 0.0219, 0.0294, 0.0604, 0.0222, 0.0207, 0.0211, 0.0188, 0.017, 0.0257, 0.0242, 0.0295, 0.0232, 0.0337, 0.028, 0.0298, 0.0209, 0.0248, 0.0307, 0.0262, 0.0284, 0.0314, 0.0312, 0.0265, 0.0214, 0.0222, 0.022, 0.0292, 0.0263, 0.0255, 0.0315, 0.0305, 0.0288, 0.0235, 0.0259, 0.021, 0.0283, 0.0243, 0.0299, 0.0283, 0.0309, 0.033, 0.0262, 0.0347, 0.0339, 0.0229, 0.0436, 0.0218, 0.0282, 0.026, 0.0257, 0.0252, 0.0262, 0.0375, 0.0385, 0.0323, 0.0295, 0.0351, 0.0353, 0.0264, 0.0382, 0.0256, 0.034, 0.0363, 0.0215, 0.0241, 0.0284, 0.0209, 0.0414, 0.0275, 0.0407, 0.0265, 0.0233, 0.0243, 0.0294, 0.0239, 0.0226, 0.028, 0.0569, 0.0266, 0.0254, 0.0283, 0.0365, 0.0251, 0.0256, 0.0313, 0.02, 0.0258, 0.0264, 0.0281, 0.0225, 0.0241, 0.0244, 0.0234, 0.026, 0.0273, 0.0226, 0.0229, 0.0352, 0.0271, 0.0308, 0.0281, 0.0221, 0.027, 0.0281, 0.0294, 0.0333, 0.0315, 0.0347, 0.0222, 0.0262, 0.0241, 0.031, 0.0232, 0.0225, 0.0222, 0.0254, 0.0214, 0.0212, 0.0254, 0.033, 0.0293, 0.0337, 0.0295, 0.0274, 0.0212, 0.0222, 0.0566, 0.0234, 0.0216, 0.0246, 0.027, 0.0267, 0.0223, 0.0228, 0.0189, 0.0253, 0.0311, 0.0206, 0.0323, 0.0267, 0.0253, 0.0309, 0.0295, 0.0292, 0.0446, 0.0381, 0.0284, 0.0268, 0.0286, 0.0241, 0.0254, 0.0326, 0.0237, 0.0222, 0.0284, 0.0245, 0.0283, 0.0307, 0.0308, 0.0302, 0.0278, 0.0251, 0.0255, 0.0279, 0.0282, 0.0321, 0.0516, 0.0234, 0.0258, 0.0224, 0.0367, 0.0262, 0.0243, 0.0305, 0.0279, 0.0322, 0.0294, 0.0379, 0.0325, 0.0315, 0.0271, 0.0368, 0.0329, 0.0289, 0.0433, 0.0337, 0.0332, 0.0232, 0.0272, 0.0421, 0.0545, 0.0307, 0.0241, 0.0191, 0.0363, 0.0217, 0.0203, 0.0221, 0.0229, 0.0216, 0.0201, 0.0276, 0.0221, 0.0306, 0.0187, 0.0225, 0.0219, 0.0241, 0.0415, 0.0218, 0.0246, 0.0428, 0.0288, 0.0199, 0.0265, 0.025, 0.0253, 0.0271, 0.0302, 0.0363, 0.0411, 0.0264, 0.0445, 0.029, 0.054, 0.019, 0.0235, 0.0206, 0.0224, 0.0214, 0.0202, 0.0221, 0.0203, 0.0236, 0.0522, 0.0463, 0.0329, 0.0389, 0.021, 0.0177, 0.0282, 0.0229, 0.0189, 0.0226, 0.0192, 0.0395, 0.0199, 0.0197, 0.0574, 0.0258, 0.0206, 0.0239, 0.0183, 0.0204, 0.021, 0.0257, 0.0209, 0.0219]

    #### 203MN
    Spatial_203MN = [0.0216, 0.0231, 0.0253, 0.0202, 0.0209, 0.0216, 0.022, 0.0215, 0.0258, 0.0227, 0.0208, 0.0252, 0.0263, 0.0216, 0.04, 0.0269, 0.0479, 0.0229, 0.0215, 0.0223, 0.025, 0.0254, 0.0221, 0.0209, 0.0282, 0.0235, 0.023, 0.0224, 0.0233, 0.0223, 0.0228, 0.0235, 0.0226, 0.023, 0.0344, 0.0282, 0.0287, 0.0294, 0.032, 0.0271, 0.0299, 0.0311, 0.0285, 0.0292, 0.0291, 0.0321, 0.0292, 0.0299, 0.0318, 0.0314, 0.0295, 0.0279, 0.0321, 0.0306, 0.0305, 0.0277, 0.027, 0.0275, 0.0316, 0.0305, 0.0295, 0.0296, 0.0284, 0.0514, 0.0338, 0.0308, 0.027, 0.028, 0.0286, 0.0305, 0.0305, 0.0298, 0.0298, 0.0298, 0.0307, 0.0318, 0.0301, 0.0281, 0.035, 0.0282, 0.0301, 0.0295, 0.0274, 0.0411, 0.0274, 0.0303, 0.0277, 0.0281, 0.0291, 0.0332, 0.021, 0.0205, 0.0216, 0.022, 0.0224, 0.0298, 0.0228, 0.0233, 0.0215, 0.0221, 0.0226, 0.022, 0.0278, 0.0527, 0.0205, 0.0204, 0.027, 0.0195, 0.0252, 0.0233, 0.0221, 0.0223, 0.0208, 0.0235, 0.0206, 0.0205, 0.0239, 0.0275, 0.0222, 0.0218, 0.0226, 0.0276, 0.0229, 0.0227, 0.0207, 0.0207, 0.0502, 0.0382, 0.0268, 0.0207, 0.0209, 0.0228, 0.0232, 0.025, 0.0268, 0.0274, 0.0272, 0.0265, 0.0269, 0.0278, 0.0254, 0.0273, 0.0264, 0.0236, 0.0275, 0.0303, 0.0275, 0.0276, 0.0294, 0.0277, 0.0252, 0.028, 0.026, 0.027, 0.0246, 0.0328, 0.0268, 0.0263, 0.031, 0.0304, 0.0301, 0.026, 0.029, 0.0242, 0.0555, 0.0602, 0.0281, 0.0237, 0.026, 0.0263, 0.0245, 0.0272, 0.0267, 0.0239, 0.0285, 0.0257, 0.0322, 0.0253, 0.0271, 0.0241, 0.0278, 0.029, 0.0232, 0.0249, 0.0282, 0.0374, 0.0289, 0.029, 0.0474, 0.0232, 0.0314, 0.0266, 0.0225, 0.0235, 0.0232, 0.0336, 0.027, 0.0237, 0.0244, 0.025, 0.0236, 0.0249, 0.0247, 0.0244, 0.0262, 0.0249, 0.0243, 0.0313, 0.0221, 0.0236, 0.024, 0.0242, 0.0253, 0.0242, 0.0267, 0.0249, 0.0297, 0.0293, 0.0269, 0.0242, 0.0275, 0.028, 0.0418, 0.024, 0.0302, 0.0226, 0.0233, 0.0243, 0.0249, 0.024, 0.0216, 0.0214, 0.0256, 0.0237, 0.0259, 0.0218, 0.0287, 0.0276, 0.0224, 0.0219, 0.0204, 0.0209, 0.0236, 0.0222, 0.0239, 0.0404, 0.0232, 0.021, 0.0218, 0.0222, 0.0217, 0.0203, 0.0214, 0.0209, 0.0231, 0.0322, 0.0225, 0.0218, 0.0563, 0.0344, 0.0439, 0.0281, 0.0242, 0.0192, 0.0208, 0.0235, 0.0256, 0.022, 0.0254, 0.022, 0.0271, 0.0224, 0.0217, 0.0433, 0.0339, 0.0237, 0.0227, 0.0257, 0.0222, 0.0239, 0.0278, 0.0274, 0.0278, 0.0257, 0.0273, 0.0281, 0.0226, 0.0224, 0.0257, 0.02, 0.0292, 0.0247, 0.03, 0.046, 0.0238, 0.0237, 0.022, 0.0221, 0.0344, 0.0241, 0.0332, 0.0357, 0.0569, 0.0313, 0.0221, 0.0198, 0.0242, 0.0211, 0.0276, 0.0253, 0.025, 0.0261, 0.0219, 0.0252, 0.0292, 0.0249, 0.0296, 0.0226, 0.0219, 0.0198, 0.0195, 0.052, 0.0203, 0.0223, 0.0233, 0.0242, 0.024, 0.0224, 0.0239, 0.0246, 0.0259, 0.0385]

    #### 204FR
    Spatial_204FR = [0.019, 0.0475, 0.0204, 0.0378, 0.044, 0.0313, 0.0177, 0.0184, 0.0187, 0.0201, 0.0213, 0.0184, 0.0414, 0.0236, 0.0243, 0.0239, 0.0221, 0.0474, 0.0346, 0.0382, 0.0443, 0.0288, 0.0259, 0.021, 0.0202, 0.0205, 0.0237, 0.0249, 0.0194, 0.0223, 0.0494, 0.0381, 0.0493, 0.0381, 0.0227, 0.0215, 0.0217, 0.0224, 0.0232, 0.0232, 0.0213, 0.0324, 0.0226, 0.0208, 0.0215, 0.028, 0.0202, 0.0257, 0.0245, 0.0268, 0.0265, 0.025, 0.028, 0.0271, 0.0302, 0.0223, 0.0249, 0.0235, 0.0237, 0.0522, 0.0272, 0.0262, 0.0259, 0.0281, 0.0277, 0.0265, 0.0242, 0.0225, 0.0296, 0.0277, 0.0265, 0.0274, 0.0281, 0.0349, 0.0272, 0.0275, 0.0574, 0.0482, 0.0397, 0.0243, 0.0244, 0.0264, 0.0261, 0.0264, 0.0383, 0.0255, 0.025, 0.0254, 0.028, 0.028, 0.0276, 0.0306, 0.0257, 0.0254, 0.0241, 0.0465, 0.0307, 0.0222, 0.0236, 0.0265, 0.0359, 0.0343, 0.0375, 0.0356, 0.035, 0.0372, 0.035, 0.0353, 0.0386, 0.0348, 0.0339, 0.0374, 0.0364, 0.0355, 0.0359, 0.0346, 0.0361, 0.0349, 0.0323, 0.0325, 0.0368, 0.0348, 0.0371, 0.0356, 0.0342, 0.0363, 0.0359, 0.0361, 0.0329, 0.0371, 0.0284, 0.0266, 0.0289, 0.0282, 0.028, 0.026, 0.0265, 0.0263, 0.0237, 0.0254, 0.0243, 0.0237, 0.0234, 0.0299, 0.0331, 0.0374, 0.029, 0.0264, 0.0275, 0.0264, 0.0276, 0.0246, 0.0266, 0.0273, 0.0602, 0.0499, 0.0244, 0.0478, 0.0264, 0.0254, 0.025, 0.0254, 0.029, 0.0276, 0.0281, 0.0252, 0.0263, 0.0273, 0.0266, 0.0279, 0.0266, 0.0294, 0.0315, 0.0254, 0.0253, 0.0246, 0.0294, 0.0263, 0.0403, 0.0454, 0.0331, 0.0275, 0.0268, 0.026, 0.0302, 0.0288, 0.0279, 0.0297, 0.0288, 0.028, 0.028, 0.0292, 0.0275, 0.0285, 0.0266, 0.0302, 0.028, 0.0253, 0.0264, 0.0244, 0.0273, 0.0245, 0.0321, 0.0272, 0.0264, 0.023, 0.0273, 0.0283, 0.0449, 0.0314, 0.0325, 0.0315, 0.028, 0.0244, 0.0342, 0.0261, 0.032, 0.0385, 0.024, 0.0262, 0.0267, 0.0275, 0.0276, 0.0306, 0.0306, 0.0283, 0.0272, 0.0258, 0.028, 0.0288, 0.03, 0.0291, 0.0284, 0.029, 0.0268, 0.0292, 0.0316, 0.0338, 0.0264, 0.029, 0.0321, 0.0268, 0.0257, 0.0272, 0.0292, 0.0276, 0.0261, 0.0267, 0.0349, 0.0356, 0.0599, 0.0491, 0.0266, 0.0246, 0.025, 0.0255, 0.0265, 0.0309, 0.0284, 0.0316, 0.0292, 0.0375, 0.0407, 0.0259, 0.0352, 0.0288, 0.0279, 0.0274, 0.0289, 0.0294, 0.0338, 0.0234, 0.0353, 0.0353, 0.0338, 0.0256, 0.0226, 0.0219, 0.022, 0.026, 0.0413, 0.0418, 0.0328, 0.0315, 0.0257, 0.0256, 0.026, 0.0261]

    #### 206FRL
    Spatial_206FRL = [0.0235, 0.0231, 0.0271, 0.0247, 0.0274, 0.0213, 0.0242, 0.03, 0.0219, 0.0232, 0.0224, 0.0227, 0.0223, 0.0231, 0.0276, 0.0221, 0.0227, 0.0228, 0.0209, 0.023, 0.023, 0.0235, 0.0261, 0.0231, 0.0209, 0.0229, 0.0229, 0.0221, 0.0217, 0.0371, 0.0312, 0.0213, 0.0224, 0.0262, 0.0261, 0.0224, 0.0215, 0.0215, 0.0212, 0.0267, 0.0225, 0.0199, 0.0265, 0.024, 0.0204, 0.0215, 0.0202, 0.0223, 0.0183, 0.0199, 0.0214, 0.0203, 0.0199, 0.0315, 0.0333, 0.0214, 0.0417, 0.0334, 0.0189, 0.0206, 0.0242, 0.0241, 0.042, 0.0265, 0.0221, 0.0209, 0.0213, 0.0236, 0.0214, 0.021, 0.0222, 0.0233, 0.0211, 0.0219, 0.0221, 0.0309, 0.0218, 0.0253, 0.0203, 0.0232, 0.0212, 0.0219, 0.0218, 0.0238, 0.0278, 0.0205, 0.0216, 0.0457, 0.0393, 0.0411, 0.0359, 0.0388, 0.0205, 0.0235, 0.0214, 0.0234, 0.0234, 0.02, 0.0197, 0.0245, 0.0234, 0.0206, 0.0255, 0.0213, 0.0243, 0.0354, 0.0236, 0.023, 0.0213, 0.0225, 0.0223, 0.0222, 0.0193, 0.0234, 0.0424, 0.0339, 0.0269, 0.0281, 0.0266, 0.0232, 0.0291, 0.0283, 0.0275, 0.0274, 0.028, 0.0286, 0.0265, 0.0326, 0.0303, 0.03, 0.0268, 0.0299, 0.0265, 0.0274, 0.0297, 0.0287, 0.0282, 0.0393, 0.0258, 0.0271, 0.0264, 0.027, 0.0266, 0.0274, 0.0281, 0.0282, 0.0294, 0.0271, 0.0273, 0.027, 0.0273, 0.0265, 0.0269, 0.0268, 0.0262, 0.0279, 0.024, 0.0271, 0.0252, 0.0261, 0.0241, 0.0262, 0.0244, 0.0241, 0.0241, 0.0267, 0.0238, 0.0246, 0.0267, 0.032, 0.022, 0.0232, 0.0235, 0.0246, 0.0251, 0.0235, 0.0244, 0.0248, 0.0246, 0.0249, 0.024, 0.0242, 0.025, 0.0244, 0.025, 0.024, 0.0263, 0.0233, 0.0241, 0.0244, 0.0366, 0.0303, 0.0298, 0.0228, 0.0238, 0.0244, 0.025, 0.0507, 0.0319, 0.0328, 0.0273, 0.0238, 0.0234, 0.0245, 0.0243, 0.0256, 0.0268, 0.0241, 0.0214, 0.0259, 0.034, 0.0238, 0.0241, 0.0276, 0.0273, 0.026, 0.026, 0.0266, 0.0329, 0.025, 0.0262, 0.0265, 0.0269, 0.0264, 0.0273, 0.0282, 0.0275, 0.0272, 0.0261, 0.026, 0.0295, 0.0292, 0.0293, 0.032, 0.0355, 0.0237, 0.0241, 0.0263, 0.0281, 0.0249, 0.0259, 0.0283, 0.0248, 0.0261, 0.0262, 0.0242, 0.0258, 0.0254, 0.0246, 0.0245, 0.0254, 0.0236, 0.0247, 0.0247, 0.0498, 0.0429, 0.0325, 0.0267, 0.0248, 0.0266, 0.0246, 0.0217, 0.0298, 0.0281, 0.0264, 0.0266, 0.0251, 0.0399, 0.0239, 0.024, 0.0268, 0.0292, 0.0284, 0.0281, 0.0281, 0.0265, 0.0275, 0.0285, 0.0254, 0.0252, 0.0251, 0.028, 0.0258, 0.0295, 0.0266, 0.0229, 0.0249, 0.025, 0.0279, 0.0262, 0.0274, 0.025, 0.0244, 0.0378, 0.0407, 0.0242, 0.0239, 0.0251, 0.0243, 0.0249, 0.0235, 0.0231, 0.025, 0.0243, 0.03, 0.0329, 0.0288, 0.0285, 0.0264, 0.0258, 0.025, 0.0277, 0.0277, 0.0261, 0.0263, 0.0255, 0.0254, 0.0327, 0.033, 0.0357, 0.0357, 0.0369, 0.0363, 0.0423, 0.0406, 0.0372, 0.0366, 0.0393, 0.0387, 0.0428, 0.0368]

    #### 211MRR
    Spatial_211MRR = [0.0261, 0.0259, 0.0271, 0.0252, 0.0256, 0.0243, 0.0273, 0.0274, 0.0251, 0.0293, 0.027, 0.0268, 0.0272, 0.0358, 0.0244, 0.0254, 0.0278, 0.0283, 0.0263, 0.0286, 0.0317, 0.028, 0.0301, 0.0345, 0.0335, 0.0281, 0.0333, 0.0296, 0.0315, 0.0273, 0.0227, 0.0276, 0.0258, 0.0249, 0.0241, 0.0251, 0.0261, 0.0276, 0.0293, 0.0309, 0.0342, 0.0282, 0.0293, 0.0258, 0.0245, 0.0234, 0.0231, 0.027, 0.0286, 0.0303, 0.0241, 0.0273, 0.0239, 0.0288, 0.0207, 0.028, 0.0237, 0.0229, 0.0236, 0.0221, 0.0264, 0.0283, 0.0351, 0.0248, 0.0295, 0.0298, 0.0282, 0.0314, 0.0251, 0.0251, 0.0254, 0.0258, 0.0279, 0.025, 0.0227, 0.0299, 0.0295, 0.0258, 0.0294, 0.0268, 0.0309, 0.0308, 0.0298, 0.0223, 0.0315, 0.0274, 0.0191, 0.0177, 0.0198, 0.0231, 0.0247, 0.0298, 0.019, 0.018, 0.0198, 0.0233, 0.0197, 0.0198, 0.0191, 0.0216, 0.022, 0.0196, 0.0223, 0.0187, 0.0206, 0.0221, 0.0222, 0.0213, 0.0182, 0.0195, 0.0181, 0.0197, 0.0208, 0.0189, 0.0278, 0.0432, 0.0298, 0.0316, 0.0335, 0.0325, 0.0226, 0.031, 0.029, 0.0293, 0.0238, 0.034, 0.0243, 0.0262, 0.0258, 0.0257, 0.0292, 0.0263, 0.037, 0.0358, 0.024, 0.0237, 0.0242, 0.0293, 0.0304, 0.0222, 0.0303, 0.0401, 0.0305, 0.0202, 0.0255, 0.0232, 0.0267, 0.028, 0.0304, 0.0263, 0.0247, 0.0247, 0.0181, 0.0195, 0.0246, 0.0219, 0.0168, 0.0183, 0.0166, 0.0176, 0.0166, 0.0207, 0.0248, 0.0179, 0.017, 0.0152, 0.0163, 0.0167, 0.0182, 0.0177, 0.0197, 0.02, 0.0197, 0.0165, 0.0205, 0.0194, 0.0186, 0.0149, 0.0218, 0.0358, 0.0159, 0.0339, 0.0169, 0.0251, 0.0239, 0.0204, 0.0236, 0.0216, 0.024, 0.0213, 0.0219, 0.0449, 0.0226, 0.0201, 0.0252, 0.0223, 0.0229, 0.0218, 0.0228, 0.0214, 0.0214, 0.0253, 0.0265, 0.026, 0.027, 0.0215, 0.0201, 0.0344, 0.0232, 0.0243, 0.0254, 0.0279, 0.03, 0.0261, 0.0267, 0.0224, 0.0233, 0.029, 0.0291, 0.0325, 0.0245, 0.023, 0.0241, 0.0205, 0.0263, 0.0381, 0.0244, 0.0281, 0.0238, 0.0227, 0.022, 0.029, 0.0238, 0.024, 0.0361, 0.0274, 0.0225, 0.0336, 0.0284, 0.026, 0.0209, 0.0225, 0.0282, 0.0316, 0.0211, 0.0245, 0.0241, 0.0247, 0.0254, 0.0304, 0.0255, 0.0338, 0.0343, 0.0337, 0.0254, 0.0343, 0.032, 0.029, 0.0264, 0.0239, 0.0274, 0.03, 0.0301, 0.045, 0.0231, 0.0209, 0.0241, 0.0251, 0.0213, 0.0222, 0.0237, 0.0246, 0.0281, 0.0272, 0.0278, 0.0294, 0.0273, 0.0233, 0.0257, 0.023, 0.0298, 0.0287, 0.0223, 0.0306, 0.0334, 0.0222, 0.0231, 0.0231, 0.0213, 0.0265, 0.0261, 0.0263, 0.0303, 0.0323, 0.0255, 0.0239]

    #### 218MN
    Spatial_218MN = [0.0419, 0.0195, 0.0167, 0.0426, 0.0182, 0.0289, 0.018, 0.0181, 0.0294, 0.0232, 0.0209, 0.0197, 0.0217, 0.0508, 0.028, 0.0226, 0.0206, 0.038, 0.023, 0.0203, 0.0207, 0.0184, 0.0191, 0.037, 0.0209, 0.0229, 0.0198, 0.0202, 0.0207, 0.0216, 0.0204, 0.0208, 0.0211, 0.0198, 0.0214, 0.036, 0.0194, 0.0195, 0.0186, 0.021, 0.0204, 0.0206, 0.0204, 0.031, 0.0216, 0.022, 0.0201, 0.0241, 0.0226, 0.0218, 0.0214, 0.0221, 0.0478, 0.0268, 0.0225, 0.0227, 0.0198, 0.0219, 0.0215, 0.0212, 0.0233, 0.0234, 0.0211, 0.0188, 0.0218, 0.022, 0.0226, 0.0225, 0.0226, 0.0217, 0.0259, 0.0245, 0.0471, 0.0216, 0.024, 0.0246, 0.0224, 0.0271, 0.0225, 0.0218, 0.0222, 0.0197, 0.026, 0.0445, 0.0243, 0.022, 0.0205, 0.0255, 0.0272, 0.0258, 0.0258, 0.0264, 0.0259, 0.031, 0.029, 0.025, 0.0259, 0.031, 0.0273, 0.0444, 0.0231, 0.0213, 0.0205, 0.0236, 0.0222, 0.022, 0.0227, 0.0216, 0.0281, 0.0252, 0.0271, 0.0271, 0.0297, 0.0293, 0.0292, 0.0251, 0.0268, 0.0268, 0.0298, 0.0317, 0.0198, 0.0184, 0.0202, 0.0186, 0.0187, 0.0229, 0.0206, 0.0207, 0.0231, 0.0184, 0.0183, 0.0219, 0.0181, 0.018, 0.0179, 0.0184, 0.0189, 0.0189, 0.02, 0.019, 0.0517, 0.0201, 0.0183, 0.0192, 0.0218, 0.0181, 0.023, 0.0285, 0.0272, 0.0223, 0.0309, 0.03, 0.0375, 0.0377, 0.0177, 0.0348, 0.0219, 0.0227, 0.0226, 0.022, 0.0224, 0.0228, 0.0236, 0.024, 0.0244, 0.0261, 0.0306, 0.0321, 0.0239, 0.0264, 0.0298, 0.0227, 0.023, 0.0291, 0.0223, 0.0237, 0.028, 0.0252, 0.0241, 0.0223, 0.0273, 0.032, 0.0281, 0.0296, 0.0386, 0.0481, 0.0248, 0.0252, 0.0263, 0.0378, 0.0339, 0.022, 0.0309, 0.0292, 0.0221, 0.0267, 0.0211, 0.0212, 0.023, 0.0213, 0.0291, 0.0327, 0.0292, 0.0232, 0.0236, 0.0252, 0.024, 0.0234, 0.0195, 0.0252, 0.0204, 0.0221, 0.0278, 0.026, 0.03, 0.0265, 0.0261, 0.0297, 0.0209, 0.0212, 0.0229, 0.0275, 0.0288, 0.0327, 0.0277, 0.0279, 0.0207, 0.0206, 0.0461, 0.0245, 0.0266, 0.0276, 0.0323, 0.0314, 0.0274, 0.0321, 0.0317, 0.0305, 0.0264, 0.0319, 0.0271, 0.0277, 0.0274, 0.0228, 0.0298, 0.0214, 0.0211, 0.0214, 0.0314, 0.0292, 0.0207, 0.0226, 0.0214, 0.0259, 0.022, 0.0198, 0.0293, 0.0229, 0.0222, 0.0318, 0.0265]

    #### 21ML
    Spatial_21ML = [0.0176, 0.0366, 0.0214, 0.0233, 0.0293, 0.0434, 0.0245, 0.0487, 0.0176, 0.0291, 0.0209, 0.0206, 0.0206, 0.0459, 0.0262, 0.0169, 0.0209, 0.0288, 0.0176, 0.0462, 0.0472, 0.0236, 0.017, 0.0245, 0.0216, 0.0175, 0.0351, 0.0167, 0.0239, 0.026, 0.0249, 0.0161, 0.0437, 0.025, 0.0359, 0.0484, 0.0367, 0.0203, 0.0493, 0.0471, 0.0349, 0.0273, 0.0198, 0.0137, 0.0216, 0.0181, 0.0263, 0.0155, 0.0184, 0.027, 0.0213, 0.018]

    Spatial = Spatial_206FRL + Spatial_204FR + Spatial_203MN + Spatial_187FN + Spatial_211MRR + Spatial_63MR + Spatial_218MN + Spatial_54MRL + Spatial_21ML
    return Spatial

def data_sigma_3():
    #### 54MRL
    Spatial_54MRL = [0.0246, 0.0264, 0.0255, 0.0486, 0.0412, 0.0416, 0.0231, 0.0357, 0.0384, 0.0557, 0.0504, 0.0352, 0.0377, 0.0448, 0.0304, 0.0451, 0.0464, 0.0238, 0.0419, 0.0341, 0.0306, 0.0258, 0.0346, 0.0308, 0.0226, 0.0294, 0.0627, 0.0521, 0.0346, 0.0278, 0.0356, 0.0584, 0.0327, 0.0384, 0.0252, 0.0332, 0.0624, 0.037, 0.0311, 0.0407, 0.0276, 0.0388, 0.0376, 0.055, 0.032, 0.0274, 0.0256, 0.0326, 0.0259, 0.042, 0.038, 0.073, 0.0294, 0.0418, 0.0441, 0.0252, 0.0259, 0.0293, 0.0418, 0.0388, 0.0281, 0.0292, 0.0424, 0.0394, 0.0451, 0.0305, 0.0361, 0.0364, 0.0325, 0.0354, 0.0389, 0.0306, 0.0532, 0.0507, 0.0393, 0.0491, 0.043, 0.0452, 0.0506, 0.0333, 0.0354, 0.031, 0.0468, 0.0339, 0.041, 0.0347, 0.0587, 0.0368, 0.0292, 0.0366, 0.0404, 0.0351, 0.0238, 0.0453, 0.0363, 0.0567, 0.0513, 0.0382, 0.0303, 0.0318, 0.0526, 0.0321, 0.0369, 0.0242, 0.0504, 0.0377, 0.0418, 0.0301, 0.0385, 0.0263, 0.0264, 0.0286, 0.0472, 0.0282, 0.0638, 0.029, 0.0259, 0.0267, 0.0392, 0.0303, 0.027, 0.0261, 0.0238, 0.053, 0.0305, 0.0303, 0.0362, 0.0244, 0.0454, 0.0267, 0.0462, 0.029, 0.0268, 0.0284, 0.0493, 0.054, 0.0291, 0.0246, 0.0333, 0.0553, 0.0413, 0.0291, 0.0295, 0.0485, 0.0464, 0.0252, 0.0388, 0.0356, 0.0398, 0.0509, 0.0277, 0.0278, 0.0342, 0.0283, 0.0458, 0.0234, 0.0281, 0.0384, 0.0301, 0.0238, 0.0311, 0.0486, 0.0474, 0.0215, 0.0282, 0.0296, 0.0282, 0.0488, 0.0595, 0.0329, 0.0398, 0.0658, 0.0313, 0.0317, 0.0262, 0.0441, 0.0361, 0.0573, 0.0335, 0.0361, 0.0243, 0.0247, 0.0569, 0.0284, 0.027, 0.0257, 0.0269, 0.0248, 0.0664, 0.0505, 0.0442, 0.0291, 0.0273, 0.0277, 0.037, 0.0413, 0.0335, 0.0423, 0.0272, 0.0525, 0.0417, 0.0468, 0.0636, 0.0417, 0.0238, 0.0284, 0.0386, 0.024, 0.0621, 0.0359, 0.024, 0.0273, 0.0304, 0.0261, 0.0641, 0.027, 0.0247, 0.0235, 0.0227, 0.0231, 0.04, 0.0384, 0.0277, 0.0526, 0.0312, 0.036, 0.0388, 0.0576, 0.0456, 0.0541, 0.0645, 0.0279, 0.0635, 0.0299, 0.0348, 0.0286, 0.0624, 0.0272, 0.0266, 0.0259, 0.0286, 0.0367, 0.0298, 0.0343, 0.0336, 0.0346, 0.0328, 0.0383, 0.0325, 0.0356, 0.052, 0.052, 0.029, 0.0266, 0.0453, 0.0347, 0.0495, 0.033, 0.0345, 0.0356, 0.036, 0.0288, 0.0382, 0.0268, 0.0266, 0.0285, 0.0411, 0.0382, 0.0257, 0.0283, 0.0273, 0.033, 0.0277, 0.0246, 0.0374, 0.0291, 0.028, 0.0405, 0.0334]

    #### 63MR
    Spatial_63MR = [0.0339, 0.0208, 0.0205, 0.0222, 0.0327, 0.0433, 0.0308, 0.0195, 0.0317, 0.0324, 0.0211, 0.0254, 0.031, 0.0225, 0.0349, 0.0277, 0.0469, 0.0233, 0.0224, 0.0488, 0.0257, 0.0258, 0.0239, 0.0353, 0.0219, 0.0329, 0.0224, 0.0213, 0.0406, 0.0421, 0.0208, 0.0247, 0.0285, 0.0304, 0.0356, 0.028, 0.0503, 0.0264, 0.0344, 0.0442, 0.0341, 0.0286, 0.027, 0.0204, 0.0258, 0.0239, 0.0248, 0.0277, 0.0367, 0.0281, 0.0233, 0.0229, 0.03, 0.0441, 0.0426, 0.0321, 0.0354, 0.0234, 0.0509, 0.027, 0.0382, 0.0318, 0.0277, 0.0472, 0.0626, 0.0265, 0.035, 0.0246, 0.0263, 0.0319, 0.033, 0.0336, 0.0289, 0.0462, 0.0461, 0.0245, 0.05, 0.0317, 0.029, 0.0341, 0.0448, 0.0337, 0.0313, 0.0252, 0.0218, 0.0192, 0.0215, 0.0419, 0.0234, 0.0203, 0.024, 0.0617, 0.0485, 0.029, 0.0387, 0.0214, 0.0479, 0.0287, 0.0372, 0.0631, 0.0344, 0.0429, 0.0345, 0.0479, 0.0352, 0.0402, 0.0422, 0.0497, 0.0344, 0.0361, 0.0402, 0.0214, 0.0246, 0.0297, 0.0289, 0.0215, 0.0287, 0.0216, 0.0321, 0.0384, 0.0421, 0.034, 0.0245, 0.0433, 0.0626, 0.0759, 0.0194, 0.0255, 0.0239, 0.021, 0.0247, 0.0302, 0.0408, 0.0547, 0.0472, 0.0279, 0.021, 0.0452, 0.0184, 0.057, 0.0366, 0.0285, 0.0553, 0.0648, 0.0306, 0.0344, 0.0221, 0.0613, 0.0295, 0.0479, 0.0288, 0.0388, 0.0489, 0.0241, 0.0301, 0.0243, 0.0388, 0.0476, 0.0439, 0.0479, 0.0417, 0.0222, 0.025, 0.0223, 0.0269, 0.0319, 0.0253, 0.0617, 0.0309, 0.0233, 0.0506, 0.0235, 0.0294, 0.0264, 0.0328, 0.0393, 0.0241, 0.0295, 0.0776, 0.0236, 0.0251, 0.03, 0.0283, 0.0268, 0.0358, 0.0238, 0.0577, 0.0373, 0.029, 0.0312, 0.024, 0.0241, 0.0254, 0.0254, 0.0279, 0.038, 0.0534, 0.0277, 0.0583, 0.0327, 0.0397, 0.0159, 0.0509, 0.0442, 0.0451, 0.0636, 0.0264, 0.0185, 0.0444, 0.0477, 0.0356]

    #### 187FN
    Spatial_187FN = [0.0296, 0.0256, 0.0293, 0.0613, 0.0423, 0.0607, 0.0271, 0.0239, 0.0232, 0.0266, 0.0237, 0.0245, 0.0234, 0.0265, 0.0269, 0.0283, 0.0243, 0.0294, 0.0265, 0.0274, 0.0261, 0.0286, 0.0245, 0.0331, 0.025, 0.0277, 0.0249, 0.0268, 0.0453, 0.025, 0.0242, 0.0267, 0.0317, 0.0227, 0.0253, 0.0264, 0.0258, 0.028, 0.0239, 0.0255, 0.029, 0.0231, 0.0244, 0.0236, 0.0362, 0.0421, 0.0351, 0.0489, 0.0402, 0.0502, 0.0361, 0.0289, 0.0458, 0.0372, 0.0329, 0.0486, 0.0323, 0.0389, 0.04, 0.0366, 0.0305, 0.0334, 0.0279, 0.0373, 0.0317, 0.0318, 0.0322, 0.0384, 0.0314, 0.0286, 0.042, 0.0415, 0.0325, 0.0389, 0.0374, 0.0432, 0.0423, 0.0381, 0.0358, 0.0354, 0.042, 0.0347, 0.0297, 0.0353, 0.0365, 0.0291, 0.0445, 0.0394, 0.0449, 0.0684, 0.0371, 0.0386, 0.0351, 0.0315, 0.0538, 0.0421, 0.0424, 0.0443, 0.0529, 0.0404, 0.0514, 0.0279, 0.0331, 0.0268, 0.0288, 0.0354, 0.0381, 0.0246, 0.0231, 0.0253, 0.0276, 0.0257, 0.0327, 0.0272, 0.0272, 0.0303, 0.0294, 0.045, 0.0216, 0.0264, 0.0324, 0.0358, 0.0254, 0.0357, 0.0296, 0.0394, 0.0376, 0.0276, 0.0383, 0.0788, 0.0275, 0.0256, 0.0259, 0.0228, 0.0203, 0.0331, 0.031, 0.0379, 0.0294, 0.044, 0.036, 0.0386, 0.0262, 0.0314, 0.0397, 0.0335, 0.0368, 0.0409, 0.0407, 0.0337, 0.0266, 0.0278, 0.0276, 0.0371, 0.0325, 0.0318, 0.0408, 0.0393, 0.0363, 0.0283, 0.0324, 0.0253, 0.0355, 0.0303, 0.038, 0.0357, 0.0395, 0.0426, 0.0334, 0.0448, 0.0436, 0.0276, 0.0561, 0.0259, 0.0359, 0.0322, 0.0318, 0.0311, 0.0329, 0.0486, 0.05, 0.0414, 0.0377, 0.0454, 0.0459, 0.0334, 0.0498, 0.0318, 0.0441, 0.0468, 0.0259, 0.0294, 0.0355, 0.0259, 0.0543, 0.0349, 0.0531, 0.0329, 0.0283, 0.0302, 0.0369, 0.0292, 0.0275, 0.0356, 0.0739, 0.0329, 0.0317, 0.0358, 0.0476, 0.0312, 0.0323, 0.0406, 0.0241, 0.0327, 0.0332, 0.0362, 0.0273, 0.0305, 0.0302, 0.029, 0.0329, 0.0344, 0.028, 0.0287, 0.0457, 0.0345, 0.0397, 0.0357, 0.0274, 0.0344, 0.036, 0.0382, 0.0432, 0.0408, 0.0451, 0.0271, 0.0327, 0.0298, 0.0399, 0.0285, 0.0278, 0.0271, 0.0318, 0.0262, 0.0261, 0.0321, 0.0429, 0.0375, 0.0438, 0.0383, 0.0349, 0.026, 0.027, 0.0738, 0.0284, 0.0263, 0.0306, 0.0341, 0.0337, 0.0274, 0.0282, 0.0226, 0.0318, 0.0401, 0.0257, 0.0411, 0.034, 0.0311, 0.0393, 0.0373, 0.0369, 0.0585, 0.0495, 0.0354, 0.0328, 0.0354, 0.0292, 0.031, 0.0415, 0.0287, 0.0269, 0.0353, 0.03, 0.0353, 0.0391, 0.0394, 0.0379, 0.0342, 0.0304, 0.0311, 0.035, 0.0354, 0.0407, 0.0664, 0.0277, 0.0313, 0.0269, 0.0467, 0.0322, 0.0299, 0.0381, 0.0354, 0.0411, 0.037, 0.0492, 0.0417, 0.04, 0.0336, 0.0476, 0.0418, 0.0368, 0.057, 0.0439, 0.0426, 0.0277, 0.0334, 0.0537, 0.07, 0.0375, 0.0297, 0.0231, 0.0463, 0.0269, 0.025, 0.0278, 0.029, 0.0269, 0.025, 0.0354, 0.0286, 0.0395, 0.0227, 0.0281, 0.0274, 0.0308, 0.0549, 0.0274, 0.0314, 0.0565, 0.0376, 0.025, 0.034, 0.0321, 0.0326, 0.0349, 0.0394, 0.047, 0.0534, 0.0337, 0.0583, 0.0367, 0.0697, 0.0231, 0.0297, 0.0255, 0.0282, 0.0267, 0.0247, 0.0275, 0.0259, 0.0298, 0.0685, 0.0598, 0.042, 0.0498, 0.0255, 0.021, 0.0352, 0.0282, 0.0234, 0.028, 0.0234, 0.0508, 0.0243, 0.0238, 0.0745, 0.0322, 0.0251, 0.0302, 0.0221, 0.025, 0.026, 0.0328, 0.0258, 0.0269]

    #### 203MN
    Spatial_203MN = [0.0253, 0.0279, 0.031, 0.0238, 0.0243, 0.0254, 0.0259, 0.0252, 0.0316, 0.0267, 0.0243, 0.031, 0.0324, 0.0253, 0.0522, 0.0329, 0.0613, 0.0275, 0.0254, 0.0263, 0.0305, 0.031, 0.0262, 0.0243, 0.0352, 0.0283, 0.0277, 0.0266, 0.0281, 0.0266, 0.0274, 0.0284, 0.0271, 0.0276, 0.0421, 0.033, 0.0337, 0.035, 0.0386, 0.0313, 0.0353, 0.0374, 0.0333, 0.0343, 0.0343, 0.0388, 0.0346, 0.0359, 0.0385, 0.0378, 0.035, 0.0327, 0.0387, 0.0365, 0.0365, 0.0323, 0.0309, 0.0315, 0.0378, 0.0365, 0.0348, 0.035, 0.0333, 0.0661, 0.0412, 0.0372, 0.0312, 0.0329, 0.0335, 0.037, 0.0366, 0.0355, 0.0354, 0.0354, 0.037, 0.0384, 0.0358, 0.0328, 0.0429, 0.0335, 0.0358, 0.0346, 0.0316, 0.0519, 0.0318, 0.0359, 0.0329, 0.033, 0.0342, 0.0421, 0.0248, 0.0241, 0.026, 0.0263, 0.0268, 0.0372, 0.0278, 0.0286, 0.0255, 0.0264, 0.0272, 0.026, 0.0345, 0.0684, 0.0243, 0.0239, 0.0336, 0.023, 0.0314, 0.0287, 0.0265, 0.0268, 0.0246, 0.0283, 0.0243, 0.024, 0.0291, 0.0341, 0.0262, 0.0261, 0.0271, 0.0346, 0.0276, 0.0274, 0.0244, 0.0244, 0.0653, 0.0479, 0.0331, 0.0242, 0.0245, 0.0275, 0.028, 0.03, 0.0327, 0.0335, 0.0332, 0.032, 0.0323, 0.0337, 0.0305, 0.0331, 0.0317, 0.0276, 0.0335, 0.038, 0.0332, 0.0334, 0.0358, 0.0335, 0.03, 0.034, 0.0315, 0.0323, 0.0288, 0.0405, 0.0323, 0.0324, 0.0387, 0.0376, 0.0373, 0.0309, 0.0353, 0.0284, 0.0704, 0.0755, 0.0339, 0.028, 0.0308, 0.0317, 0.0291, 0.0329, 0.0324, 0.0282, 0.035, 0.0306, 0.0404, 0.0304, 0.0329, 0.0282, 0.034, 0.0357, 0.0274, 0.0301, 0.0352, 0.0486, 0.0361, 0.0361, 0.0621, 0.0274, 0.0388, 0.0318, 0.0265, 0.028, 0.0274, 0.0422, 0.0327, 0.0278, 0.0291, 0.0298, 0.0277, 0.0296, 0.0296, 0.029, 0.0317, 0.0297, 0.0287, 0.0388, 0.0258, 0.0275, 0.0284, 0.0284, 0.03, 0.0285, 0.0323, 0.0299, 0.0371, 0.0366, 0.0327, 0.0286, 0.0335, 0.0343, 0.0533, 0.0278, 0.0375, 0.0266, 0.0277, 0.0291, 0.0294, 0.0293, 0.0259, 0.0259, 0.0315, 0.0288, 0.0323, 0.0261, 0.0358, 0.0342, 0.0272, 0.0265, 0.0241, 0.0251, 0.0288, 0.0267, 0.0292, 0.0512, 0.0278, 0.0251, 0.0264, 0.0267, 0.0263, 0.0241, 0.0254, 0.025, 0.0281, 0.041, 0.0272, 0.0261, 0.0732, 0.0437, 0.055, 0.0344, 0.0295, 0.0224, 0.0245, 0.0288, 0.0314, 0.0267, 0.0317, 0.0264, 0.0341, 0.0269, 0.0265, 0.0563, 0.0427, 0.0288, 0.0275, 0.0317, 0.0266, 0.0292, 0.0349, 0.0343, 0.0346, 0.0316, 0.0342, 0.0353, 0.0271, 0.0271, 0.0315, 0.0235, 0.0368, 0.0302, 0.0379, 0.0593, 0.0288, 0.0286, 0.0263, 0.0263, 0.0435, 0.029, 0.0419, 0.0453, 0.073, 0.0389, 0.0267, 0.0234, 0.0298, 0.0257, 0.0349, 0.0309, 0.0311, 0.0325, 0.0275, 0.0322, 0.0373, 0.0313, 0.0381, 0.0277, 0.0266, 0.0236, 0.0233, 0.0678, 0.0247, 0.0276, 0.0292, 0.0301, 0.0299, 0.0276, 0.0299, 0.0309, 0.0326, 0.0489]

    #### 204FR
    Spatial_204FR = [0.0226, 0.0605, 0.0248, 0.0486, 0.0555, 0.04, 0.0211, 0.022, 0.0226, 0.0248, 0.0265, 0.0221, 0.0521, 0.0293, 0.0301, 0.0296, 0.0268, 0.0609, 0.0422, 0.0469, 0.0565, 0.0356, 0.0319, 0.0249, 0.0239, 0.0244, 0.0289, 0.0309, 0.0226, 0.0267, 0.0623, 0.0458, 0.0615, 0.0474, 0.0273, 0.0256, 0.0264, 0.0268, 0.028, 0.0282, 0.0252, 0.0408, 0.0276, 0.0247, 0.0258, 0.0352, 0.0238, 0.0308, 0.0293, 0.0323, 0.0318, 0.0297, 0.0337, 0.0326, 0.0369, 0.026, 0.0297, 0.0278, 0.0281, 0.067, 0.0328, 0.0314, 0.0312, 0.0341, 0.0335, 0.032, 0.0291, 0.0266, 0.0366, 0.0336, 0.032, 0.0334, 0.0341, 0.0434, 0.0327, 0.0333, 0.0726, 0.0607, 0.0498, 0.029, 0.029, 0.0319, 0.032, 0.032, 0.048, 0.0307, 0.03, 0.031, 0.0344, 0.0344, 0.0342, 0.0375, 0.0308, 0.0307, 0.0287, 0.0582, 0.037, 0.0259, 0.0277, 0.032, 0.0431, 0.0407, 0.0455, 0.0424, 0.0417, 0.045, 0.042, 0.0421, 0.0471, 0.0416, 0.0401, 0.0457, 0.0442, 0.0428, 0.0431, 0.0416, 0.0435, 0.0419, 0.0374, 0.0381, 0.0445, 0.0411, 0.0448, 0.0433, 0.0412, 0.0439, 0.0433, 0.0435, 0.0389, 0.0458, 0.0347, 0.0324, 0.0356, 0.0343, 0.0339, 0.0316, 0.0318, 0.0314, 0.0278, 0.0302, 0.0289, 0.0276, 0.0274, 0.0373, 0.0416, 0.0477, 0.0347, 0.0314, 0.0334, 0.0318, 0.0339, 0.0291, 0.032, 0.0329, 0.0749, 0.0618, 0.0286, 0.0605, 0.0318, 0.0304, 0.0296, 0.0305, 0.0358, 0.0332, 0.0343, 0.0303, 0.032, 0.0333, 0.0324, 0.0343, 0.0324, 0.0367, 0.0383, 0.0302, 0.0302, 0.0292, 0.0357, 0.0309, 0.0504, 0.0573, 0.0394, 0.033, 0.032, 0.031, 0.0367, 0.0353, 0.0342, 0.0364, 0.0351, 0.0337, 0.0339, 0.0361, 0.0329, 0.0345, 0.0316, 0.0371, 0.0329, 0.0296, 0.0314, 0.0282, 0.0326, 0.0286, 0.039, 0.0321, 0.0311, 0.0264, 0.0327, 0.0347, 0.056, 0.0381, 0.0396, 0.0381, 0.0337, 0.0283, 0.0423, 0.0304, 0.039, 0.048, 0.0275, 0.0312, 0.0317, 0.0331, 0.0335, 0.0375, 0.0374, 0.0347, 0.0327, 0.0302, 0.0342, 0.0353, 0.037, 0.0355, 0.0345, 0.0347, 0.0314, 0.0356, 0.0395, 0.0426, 0.0315, 0.0357, 0.0392, 0.0316, 0.0299, 0.0323, 0.0356, 0.0337, 0.0308, 0.0317, 0.0434, 0.044, 0.0739, 0.0603, 0.0311, 0.0283, 0.0294, 0.0301, 0.0316, 0.0381, 0.0346, 0.0381, 0.035, 0.0462, 0.0504, 0.03, 0.043, 0.0347, 0.0333, 0.0327, 0.0369, 0.0374, 0.044, 0.0283, 0.0443, 0.0437, 0.0412, 0.0312, 0.0269, 0.026, 0.0261, 0.032, 0.0517, 0.0522, 0.0416, 0.0394, 0.0316, 0.0316, 0.0323, 0.0326]

    #### 206FRL
    Spatial_206FRL = [0.0285, 0.028, 0.0339, 0.03, 0.0343, 0.0254, 0.0296, 0.0382, 0.026, 0.0282, 0.0268, 0.0273, 0.0266, 0.0276, 0.0347, 0.0265, 0.0274, 0.0274, 0.0244, 0.0279, 0.0276, 0.0283, 0.0322, 0.0278, 0.0243, 0.0274, 0.0273, 0.0262, 0.0258, 0.0482, 0.0396, 0.0251, 0.0265, 0.0325, 0.0326, 0.0266, 0.0252, 0.0252, 0.0257, 0.0337, 0.0274, 0.0235, 0.0333, 0.0297, 0.0243, 0.0265, 0.024, 0.0269, 0.0215, 0.0238, 0.0259, 0.0243, 0.0237, 0.0406, 0.0432, 0.0259, 0.0536, 0.043, 0.0225, 0.025, 0.0302, 0.0299, 0.0547, 0.0336, 0.0269, 0.0253, 0.0257, 0.0291, 0.0258, 0.0254, 0.0271, 0.0286, 0.0256, 0.0267, 0.0272, 0.0397, 0.0266, 0.0316, 0.0245, 0.0288, 0.026, 0.0267, 0.0266, 0.0296, 0.0351, 0.0251, 0.0265, 0.0598, 0.051, 0.0535, 0.0461, 0.0502, 0.025, 0.0292, 0.0261, 0.029, 0.0289, 0.0245, 0.0236, 0.0305, 0.0289, 0.0251, 0.0322, 0.0261, 0.0305, 0.0455, 0.0292, 0.0283, 0.0262, 0.0279, 0.0274, 0.0271, 0.0234, 0.0289, 0.0546, 0.0423, 0.032, 0.0339, 0.0318, 0.0272, 0.0353, 0.0339, 0.0332, 0.033, 0.0345, 0.0346, 0.0318, 0.0405, 0.037, 0.0367, 0.032, 0.0365, 0.0317, 0.0329, 0.0368, 0.0348, 0.0341, 0.05, 0.0303, 0.0322, 0.0313, 0.0322, 0.0317, 0.0327, 0.0337, 0.0337, 0.0351, 0.0327, 0.0327, 0.0323, 0.0324, 0.0313, 0.032, 0.0321, 0.0312, 0.0331, 0.0277, 0.032, 0.0296, 0.0304, 0.0284, 0.032, 0.029, 0.0286, 0.0285, 0.0323, 0.0286, 0.0295, 0.0323, 0.04, 0.0254, 0.027, 0.0275, 0.0295, 0.03, 0.0279, 0.0291, 0.0298, 0.0295, 0.0298, 0.0286, 0.0285, 0.0295, 0.029, 0.0295, 0.0286, 0.0319, 0.0275, 0.0285, 0.0292, 0.0468, 0.0374, 0.0369, 0.0269, 0.0283, 0.029, 0.0303, 0.0659, 0.04, 0.0414, 0.0336, 0.0282, 0.0276, 0.0293, 0.0292, 0.0311, 0.0327, 0.0285, 0.0249, 0.0312, 0.0428, 0.028, 0.0289, 0.0334, 0.0331, 0.0313, 0.0316, 0.0327, 0.0413, 0.0295, 0.0314, 0.0317, 0.0326, 0.0316, 0.033, 0.0347, 0.0335, 0.033, 0.0314, 0.031, 0.0366, 0.036, 0.0361, 0.0404, 0.0454, 0.0278, 0.0287, 0.0317, 0.034, 0.0296, 0.0314, 0.0345, 0.0294, 0.0313, 0.0316, 0.0282, 0.0306, 0.0301, 0.0289, 0.0291, 0.0302, 0.0277, 0.0293, 0.0291, 0.0652, 0.0553, 0.0408, 0.0324, 0.0294, 0.0318, 0.0294, 0.0252, 0.0368, 0.0342, 0.032, 0.0321, 0.03, 0.0511, 0.0283, 0.0288, 0.0334, 0.036, 0.0351, 0.035, 0.0344, 0.0324, 0.0341, 0.0347, 0.0304, 0.0304, 0.0301, 0.0348, 0.0308, 0.0367, 0.0321, 0.0268, 0.0295, 0.0299, 0.0341, 0.0317, 0.0336, 0.0299, 0.0285, 0.0488, 0.0521, 0.0286, 0.0282, 0.0301, 0.0287, 0.0297, 0.0278, 0.0269, 0.0298, 0.0287, 0.0375, 0.0415, 0.0354, 0.0349, 0.0326, 0.0314, 0.03, 0.0346, 0.0347, 0.0319, 0.0319, 0.0308, 0.0307, 0.0377, 0.0382, 0.0423, 0.0431, 0.0448, 0.0443, 0.0526, 0.0499, 0.045, 0.044, 0.0484, 0.0474, 0.0528, 0.0442]

    #### 211MRR
    Spatial_211MRR = [0.0316, 0.031, 0.0334, 0.0303, 0.0312, 0.0286, 0.0334, 0.034, 0.0298, 0.0366, 0.0331, 0.0327, 0.0335, 0.0452, 0.0283, 0.0303, 0.0342, 0.0353, 0.0318, 0.0354, 0.0404, 0.0346, 0.0378, 0.0442, 0.0435, 0.0354, 0.0426, 0.0366, 0.0389, 0.0344, 0.026, 0.0331, 0.0306, 0.0292, 0.0284, 0.0299, 0.0314, 0.0339, 0.0365, 0.0389, 0.0448, 0.0352, 0.0371, 0.0311, 0.0308, 0.0291, 0.0286, 0.0345, 0.0371, 0.0399, 0.0302, 0.0347, 0.0298, 0.0371, 0.0254, 0.0354, 0.0294, 0.0284, 0.0288, 0.0271, 0.0344, 0.0363, 0.0464, 0.031, 0.0384, 0.0388, 0.0362, 0.0408, 0.0323, 0.0319, 0.0316, 0.0327, 0.0363, 0.0323, 0.0289, 0.0386, 0.0384, 0.033, 0.0377, 0.0342, 0.0408, 0.0401, 0.0384, 0.0283, 0.0417, 0.0354, 0.0239, 0.0215, 0.0253, 0.03, 0.0323, 0.0396, 0.0236, 0.0219, 0.0243, 0.0301, 0.0246, 0.0249, 0.024, 0.0274, 0.0285, 0.0244, 0.0284, 0.0237, 0.0259, 0.0291, 0.028, 0.0267, 0.0221, 0.0244, 0.0225, 0.0249, 0.0265, 0.0234, 0.0367, 0.0569, 0.0381, 0.0411, 0.0432, 0.0416, 0.0269, 0.04, 0.037, 0.0376, 0.0298, 0.0428, 0.0291, 0.0326, 0.0323, 0.0318, 0.0374, 0.0325, 0.049, 0.0466, 0.0289, 0.0284, 0.0297, 0.0375, 0.0384, 0.0263, 0.0378, 0.0511, 0.0381, 0.0237, 0.0315, 0.028, 0.0332, 0.0348, 0.0376, 0.0327, 0.0302, 0.0302, 0.0231, 0.0252, 0.033, 0.0281, 0.021, 0.0238, 0.021, 0.0227, 0.0211, 0.0273, 0.0329, 0.0232, 0.0213, 0.0184, 0.0204, 0.0211, 0.0232, 0.0227, 0.0254, 0.0258, 0.0256, 0.0209, 0.0271, 0.0255, 0.024, 0.0185, 0.0285, 0.0481, 0.0192, 0.0431, 0.0206, 0.0327, 0.0299, 0.0255, 0.0301, 0.0277, 0.0296, 0.0254, 0.0267, 0.0577, 0.0275, 0.0237, 0.0313, 0.0268, 0.029, 0.0264, 0.0283, 0.0262, 0.0267, 0.0325, 0.0339, 0.0333, 0.0345, 0.0264, 0.0237, 0.0443, 0.0276, 0.0308, 0.0329, 0.0358, 0.0391, 0.0332, 0.0343, 0.0277, 0.0293, 0.0372, 0.0379, 0.0423, 0.0308, 0.0284, 0.03, 0.0251, 0.0334, 0.0512, 0.0315, 0.0357, 0.0296, 0.0282, 0.0268, 0.0379, 0.0301, 0.0302, 0.0471, 0.0354, 0.0282, 0.0444, 0.0367, 0.0324, 0.0252, 0.0283, 0.0364, 0.0401, 0.025, 0.0309, 0.0308, 0.0312, 0.0323, 0.0396, 0.0325, 0.0448, 0.0454, 0.0442, 0.0318, 0.0453, 0.0414, 0.0375, 0.0341, 0.0312, 0.0352, 0.0392, 0.0383, 0.0579, 0.0281, 0.0253, 0.0304, 0.0319, 0.0267, 0.0274, 0.0296, 0.0315, 0.037, 0.0349, 0.0365, 0.0388, 0.036, 0.0294, 0.0331, 0.0293, 0.0394, 0.0375, 0.0277, 0.0402, 0.043, 0.0273, 0.029, 0.0291, 0.0257, 0.0347, 0.0329, 0.0333, 0.0399, 0.0426, 0.033, 0.0304]

    #### 218MN
    Spatial_218MN = [0.0548, 0.0242, 0.0202, 0.0552, 0.0224, 0.0371, 0.0222, 0.0224, 0.0383, 0.0287, 0.026, 0.0236, 0.0268, 0.0649, 0.0354, 0.0282, 0.0245, 0.0489, 0.0286, 0.0246, 0.0251, 0.0222, 0.0227, 0.0473, 0.0262, 0.0284, 0.0241, 0.0246, 0.0253, 0.0264, 0.0246, 0.0253, 0.026, 0.0241, 0.0266, 0.046, 0.0233, 0.0241, 0.0223, 0.0257, 0.025, 0.025, 0.0243, 0.0387, 0.0267, 0.027, 0.0246, 0.0299, 0.0279, 0.0267, 0.026, 0.0269, 0.0606, 0.033, 0.0275, 0.0274, 0.0235, 0.0265, 0.026, 0.0257, 0.0287, 0.0288, 0.0257, 0.0222, 0.0263, 0.027, 0.0276, 0.0276, 0.028, 0.026, 0.033, 0.0307, 0.0599, 0.0261, 0.0299, 0.0307, 0.0274, 0.0346, 0.0277, 0.0267, 0.0267, 0.0231, 0.0323, 0.0568, 0.0306, 0.0267, 0.0245, 0.0326, 0.0348, 0.0327, 0.0326, 0.0338, 0.033, 0.0406, 0.0369, 0.0316, 0.033, 0.0405, 0.0347, 0.0561, 0.0284, 0.0262, 0.0255, 0.0293, 0.0271, 0.0269, 0.0282, 0.0267, 0.0364, 0.0321, 0.0347, 0.035, 0.0385, 0.0388, 0.0378, 0.0318, 0.0348, 0.0349, 0.039, 0.0409, 0.0253, 0.0228, 0.0251, 0.023, 0.0231, 0.0298, 0.0266, 0.0261, 0.0293, 0.0222, 0.0223, 0.0271, 0.0222, 0.022, 0.0219, 0.0225, 0.0236, 0.0235, 0.025, 0.0236, 0.0664, 0.0248, 0.0228, 0.024, 0.0279, 0.0225, 0.03, 0.0376, 0.0355, 0.0282, 0.0396, 0.0389, 0.0479, 0.0491, 0.0211, 0.0447, 0.026, 0.0273, 0.0278, 0.0269, 0.0269, 0.0278, 0.0294, 0.0299, 0.0306, 0.033, 0.0397, 0.0415, 0.0302, 0.0336, 0.038, 0.0276, 0.0282, 0.0364, 0.0274, 0.0298, 0.0362, 0.0314, 0.0298, 0.0269, 0.0352, 0.0419, 0.0358, 0.0384, 0.049, 0.0612, 0.0317, 0.0319, 0.0343, 0.0502, 0.0445, 0.0272, 0.0405, 0.038, 0.0278, 0.0338, 0.0258, 0.0264, 0.0292, 0.0266, 0.0384, 0.0432, 0.0386, 0.0296, 0.0296, 0.0325, 0.0304, 0.0296, 0.0238, 0.0321, 0.0252, 0.0277, 0.0364, 0.0338, 0.0394, 0.0349, 0.0345, 0.0384, 0.0257, 0.0262, 0.0295, 0.0361, 0.0381, 0.0424, 0.0357, 0.0358, 0.0258, 0.0256, 0.0588, 0.0318, 0.035, 0.0363, 0.0426, 0.0415, 0.0358, 0.042, 0.0416, 0.04, 0.0346, 0.0418, 0.0355, 0.0364, 0.0363, 0.0291, 0.0389, 0.0267, 0.0266, 0.0282, 0.0419, 0.0384, 0.0261, 0.0287, 0.0275, 0.0338, 0.0284, 0.0247, 0.0383, 0.0293, 0.0284, 0.0408, 0.0331]

    #### 21ML
    Spatial_21ML = [0.0221, 0.0476, 0.0266, 0.0305, 0.0378, 0.0562, 0.0306, 0.0635, 0.0224, 0.0375, 0.0263, 0.0262, 0.026, 0.0598, 0.0334, 0.0203, 0.0262, 0.0369, 0.0214, 0.0609, 0.0608, 0.0299, 0.0205, 0.0311, 0.0268, 0.0211, 0.0453, 0.0201, 0.0302, 0.0335, 0.0319, 0.0192, 0.0576, 0.0326, 0.0463, 0.0625, 0.0462, 0.0254, 0.0633, 0.0606, 0.0449, 0.0343, 0.0249, 0.0168, 0.0275, 0.0229, 0.034, 0.0194, 0.0234, 0.0347, 0.0272, 0.0226]

    Spatial = Spatial_206FRL + Spatial_204FR + Spatial_203MN + Spatial_187FN + Spatial_211MRR + Spatial_63MR + Spatial_218MN + Spatial_54MRL + Spatial_21ML
    return Spatial

def data_sigma_4():
    #### 54MRL
    Spatial_54MRL = [0.0286, 0.0306, 0.0297, 0.0573, 0.0479, 0.0489, 0.0267, 0.0421, 0.0449, 0.0652, 0.0596, 0.0411, 0.0444, 0.0518, 0.0354, 0.0529, 0.0539, 0.0277, 0.0495, 0.0404, 0.0353, 0.0293, 0.0407, 0.0357, 0.0259, 0.034, 0.0731, 0.0601, 0.0402, 0.0316, 0.0419, 0.0684, 0.0371, 0.0435, 0.0285, 0.0383, 0.0708, 0.0413, 0.0361, 0.048, 0.032, 0.0453, 0.0433, 0.0648, 0.0372, 0.0322, 0.0294, 0.0381, 0.0298, 0.0509, 0.0437, 0.0852, 0.0338, 0.0473, 0.051, 0.0288, 0.0301, 0.0345, 0.0485, 0.0452, 0.0319, 0.0336, 0.0488, 0.0468, 0.0531, 0.0345, 0.0422, 0.0422, 0.0382, 0.0413, 0.0454, 0.035, 0.0616, 0.0596, 0.046, 0.0579, 0.0498, 0.0525, 0.0589, 0.0383, 0.0408, 0.0353, 0.0541, 0.0393, 0.0489, 0.0405, 0.0676, 0.042, 0.0335, 0.0424, 0.0477, 0.0405, 0.0266, 0.0526, 0.042, 0.0669, 0.0598, 0.0452, 0.0345, 0.0368, 0.0614, 0.0355, 0.0427, 0.0273, 0.0583, 0.0418, 0.047, 0.0347, 0.0446, 0.0299, 0.0298, 0.0327, 0.055, 0.032, 0.0749, 0.0335, 0.0297, 0.0305, 0.0457, 0.0352, 0.0307, 0.0297, 0.0271, 0.0621, 0.0351, 0.0345, 0.0422, 0.0279, 0.0524, 0.0308, 0.0529, 0.0337, 0.0308, 0.0325, 0.0571, 0.062, 0.0335, 0.0279, 0.0383, 0.0645, 0.0473, 0.0336, 0.0335, 0.0558, 0.0548, 0.0283, 0.0449, 0.041, 0.0462, 0.0592, 0.0312, 0.0318, 0.0396, 0.0317, 0.0524, 0.0266, 0.0321, 0.0453, 0.0342, 0.0268, 0.0368, 0.0558, 0.0554, 0.0241, 0.0325, 0.0337, 0.0327, 0.0573, 0.0687, 0.0375, 0.0456, 0.0756, 0.0362, 0.0367, 0.0296, 0.0513, 0.0419, 0.0666, 0.0393, 0.0422, 0.0277, 0.028, 0.0655, 0.0322, 0.0311, 0.0291, 0.0308, 0.0283, 0.0766, 0.0578, 0.0506, 0.0342, 0.0316, 0.0322, 0.0431, 0.048, 0.0391, 0.0488, 0.0311, 0.0627, 0.0478, 0.0537, 0.0721, 0.0472, 0.0272, 0.0324, 0.0445, 0.0272, 0.0709, 0.04, 0.0271, 0.0318, 0.0355, 0.0299, 0.0762, 0.0315, 0.0281, 0.0266, 0.0258, 0.0264, 0.0478, 0.0446, 0.0319, 0.0608, 0.0361, 0.0426, 0.0446, 0.0668, 0.0524, 0.0621, 0.074, 0.0316, 0.0724, 0.0344, 0.0395, 0.0326, 0.0721, 0.0307, 0.0307, 0.0296, 0.033, 0.0428, 0.0345, 0.0395, 0.0386, 0.0399, 0.0385, 0.0448, 0.0382, 0.0407, 0.0596, 0.0597, 0.0336, 0.0307, 0.0526, 0.0405, 0.057, 0.0381, 0.0421, 0.0433, 0.0443, 0.0347, 0.0466, 0.0318, 0.0319, 0.0356, 0.0508, 0.0461, 0.0305, 0.0341, 0.0335, 0.0405, 0.0336, 0.029, 0.0452, 0.0351, 0.0338, 0.0477, 0.0385]

    #### 63MR
    Spatial_63MR = [0.0397, 0.024, 0.0236, 0.0258, 0.0383, 0.0508, 0.0361, 0.0222, 0.0369, 0.0373, 0.0242, 0.0298, 0.0364, 0.0258, 0.0409, 0.0327, 0.056, 0.0268, 0.0258, 0.0581, 0.0299, 0.0296, 0.0276, 0.0408, 0.025, 0.0381, 0.0257, 0.0244, 0.0487, 0.0503, 0.0239, 0.0283, 0.0325, 0.0346, 0.0422, 0.0318, 0.0592, 0.0299, 0.0402, 0.0522, 0.0393, 0.033, 0.0308, 0.0227, 0.0292, 0.027, 0.028, 0.0322, 0.0426, 0.0322, 0.0261, 0.0259, 0.0346, 0.0513, 0.0497, 0.0367, 0.0406, 0.0263, 0.0597, 0.0306, 0.0439, 0.0363, 0.0317, 0.0547, 0.073, 0.0298, 0.0411, 0.0283, 0.0303, 0.0374, 0.039, 0.0383, 0.0326, 0.0532, 0.0531, 0.0287, 0.059, 0.0362, 0.0335, 0.0393, 0.052, 0.0393, 0.0367, 0.0289, 0.0246, 0.0222, 0.0243, 0.0485, 0.0267, 0.0236, 0.0276, 0.0711, 0.0566, 0.0335, 0.046, 0.0241, 0.0548, 0.033, 0.043, 0.073, 0.0389, 0.0496, 0.0402, 0.0568, 0.0406, 0.0464, 0.0493, 0.0571, 0.0393, 0.0412, 0.0468, 0.0241, 0.028, 0.0341, 0.0331, 0.0244, 0.0333, 0.0245, 0.0369, 0.0449, 0.0493, 0.0405, 0.028, 0.0507, 0.072, 0.0873, 0.0218, 0.0299, 0.0277, 0.0242, 0.0286, 0.0348, 0.0469, 0.0628, 0.0549, 0.0322, 0.0241, 0.0526, 0.0212, 0.0662, 0.0425, 0.0337, 0.0637, 0.075, 0.0356, 0.0396, 0.0251, 0.071, 0.034, 0.055, 0.0332, 0.0446, 0.0562, 0.0279, 0.0351, 0.0279, 0.0455, 0.0559, 0.0507, 0.0548, 0.0477, 0.0259, 0.0289, 0.0252, 0.0311, 0.0375, 0.0289, 0.0716, 0.0353, 0.0263, 0.058, 0.027, 0.0337, 0.0306, 0.0371, 0.0455, 0.0279, 0.034, 0.0888, 0.0271, 0.0288, 0.0348, 0.0325, 0.0305, 0.0414, 0.0268, 0.0668, 0.042, 0.033, 0.0351, 0.0273, 0.0272, 0.029, 0.0294, 0.032, 0.0437, 0.0612, 0.0315, 0.067, 0.0376, 0.0457, 0.0179, 0.0586, 0.0505, 0.0512, 0.0742, 0.0311, 0.0213, 0.0516, 0.0552, 0.0411]

    #### 187FN
    Spatial_187FN = [0.0351, 0.0297, 0.0347, 0.0722, 0.0492, 0.0704, 0.0319, 0.0278, 0.0268, 0.0313, 0.0275, 0.0289, 0.0275, 0.0313, 0.0318, 0.0333, 0.0283, 0.0349, 0.0313, 0.0324, 0.0305, 0.0335, 0.029, 0.0397, 0.0297, 0.0327, 0.0291, 0.0315, 0.0529, 0.0289, 0.0279, 0.0312, 0.0371, 0.026, 0.0293, 0.0312, 0.0301, 0.0325, 0.0277, 0.0297, 0.0342, 0.0269, 0.0284, 0.0275, 0.0427, 0.0501, 0.0414, 0.059, 0.0476, 0.0601, 0.0423, 0.0333, 0.0543, 0.0432, 0.0382, 0.0564, 0.0374, 0.045, 0.0471, 0.0428, 0.035, 0.0389, 0.0319, 0.0439, 0.0366, 0.0369, 0.0376, 0.0453, 0.0361, 0.0334, 0.0498, 0.0483, 0.0374, 0.0459, 0.0439, 0.0515, 0.0501, 0.0452, 0.0424, 0.0415, 0.05, 0.0407, 0.0342, 0.0413, 0.0429, 0.0332, 0.0529, 0.0461, 0.0537, 0.0803, 0.0427, 0.0448, 0.0412, 0.0368, 0.0641, 0.0499, 0.0506, 0.0531, 0.0635, 0.0487, 0.0619, 0.033, 0.0394, 0.0318, 0.0341, 0.0427, 0.0457, 0.0287, 0.0266, 0.0298, 0.0328, 0.0304, 0.0388, 0.0322, 0.0322, 0.0359, 0.0353, 0.0525, 0.0248, 0.0313, 0.0388, 0.0425, 0.0299, 0.0428, 0.0355, 0.047, 0.0456, 0.0329, 0.0462, 0.092, 0.0322, 0.0302, 0.0306, 0.0263, 0.0233, 0.0401, 0.0374, 0.0451, 0.0351, 0.0533, 0.0432, 0.0465, 0.0312, 0.0373, 0.0479, 0.0401, 0.0446, 0.0497, 0.0493, 0.04, 0.0314, 0.0331, 0.0327, 0.0442, 0.0383, 0.0377, 0.0494, 0.0467, 0.0431, 0.0328, 0.0383, 0.0293, 0.042, 0.0358, 0.0451, 0.0421, 0.0475, 0.0514, 0.0398, 0.0535, 0.052, 0.032, 0.0647, 0.0294, 0.0426, 0.0374, 0.037, 0.0359, 0.0391, 0.0584, 0.0601, 0.0498, 0.045, 0.0544, 0.0555, 0.0404, 0.0599, 0.0374, 0.0532, 0.0559, 0.03, 0.0342, 0.0417, 0.0306, 0.0655, 0.0415, 0.0633, 0.0384, 0.0331, 0.0357, 0.0437, 0.0341, 0.0321, 0.0429, 0.0862, 0.0389, 0.0376, 0.0426, 0.0576, 0.0368, 0.0385, 0.0489, 0.028, 0.0389, 0.0393, 0.0436, 0.0315, 0.0362, 0.0351, 0.034, 0.0392, 0.0407, 0.0331, 0.0335, 0.0548, 0.041, 0.0473, 0.0424, 0.0324, 0.0413, 0.0429, 0.0466, 0.0522, 0.0491, 0.0537, 0.0312, 0.0383, 0.0351, 0.0482, 0.0334, 0.0327, 0.0313, 0.0375, 0.0308, 0.0308, 0.0383, 0.0512, 0.0449, 0.053, 0.0465, 0.0415, 0.0305, 0.0314, 0.0863, 0.0328, 0.0307, 0.0358, 0.0406, 0.0394, 0.0321, 0.033, 0.0258, 0.0379, 0.0483, 0.0308, 0.0494, 0.0412, 0.0367, 0.0471, 0.0444, 0.044, 0.0712, 0.06, 0.0417, 0.0379, 0.0416, 0.0341, 0.0362, 0.0496, 0.0335, 0.0312, 0.0416, 0.0352, 0.0417, 0.0467, 0.0473, 0.0447, 0.0399, 0.0352, 0.0363, 0.0415, 0.042, 0.0484, 0.077, 0.0315, 0.0362, 0.0312, 0.0553, 0.0379, 0.0352, 0.0449, 0.0425, 0.0492, 0.044, 0.0594, 0.0501, 0.0476, 0.0397, 0.0571, 0.0497, 0.0443, 0.0688, 0.0535, 0.0511, 0.0321, 0.0388, 0.0624, 0.081, 0.0427, 0.0338, 0.0268, 0.0534, 0.0313, 0.0294, 0.0331, 0.0347, 0.0319, 0.0295, 0.0423, 0.0346, 0.0473, 0.0263, 0.0331, 0.0321, 0.0367, 0.0662, 0.0327, 0.0376, 0.0683, 0.0456, 0.0297, 0.041, 0.0388, 0.0392, 0.0418, 0.0474, 0.0552, 0.0629, 0.0397, 0.0688, 0.0419, 0.0805, 0.027, 0.0353, 0.0302, 0.0334, 0.0318, 0.0288, 0.0323, 0.031, 0.0356, 0.0807, 0.0694, 0.0492, 0.0575, 0.029, 0.024, 0.0402, 0.0324, 0.0276, 0.0327, 0.0271, 0.0592, 0.0283, 0.0276, 0.0862, 0.0375, 0.0291, 0.036, 0.0258, 0.0291, 0.0308, 0.0396, 0.0302, 0.0314]

    #### 203MN
    Spatial_203MN = [0.0287, 0.0323, 0.0362, 0.0271, 0.0275, 0.0289, 0.0295, 0.0287, 0.0368, 0.0303, 0.0275, 0.0362, 0.0381, 0.0288, 0.0634, 0.0377, 0.0705, 0.0318, 0.0294, 0.0299, 0.0354, 0.0356, 0.0301, 0.0276, 0.0413, 0.0327, 0.0324, 0.0305, 0.0324, 0.0305, 0.0315, 0.0328, 0.0312, 0.0318, 0.0489, 0.0372, 0.0379, 0.0403, 0.0447, 0.0353, 0.0402, 0.0433, 0.0378, 0.039, 0.039, 0.0448, 0.0395, 0.0412, 0.0445, 0.044, 0.0402, 0.0372, 0.0445, 0.0419, 0.0421, 0.0366, 0.0345, 0.0352, 0.0432, 0.042, 0.0397, 0.04, 0.038, 0.0773, 0.0467, 0.0426, 0.0349, 0.0375, 0.0381, 0.0432, 0.0426, 0.0411, 0.0406, 0.0402, 0.0428, 0.0445, 0.0409, 0.037, 0.05, 0.0382, 0.0409, 0.0392, 0.0354, 0.0609, 0.0354, 0.0408, 0.0377, 0.0377, 0.0389, 0.0488, 0.0284, 0.0275, 0.0301, 0.0303, 0.0308, 0.043, 0.0325, 0.0339, 0.0292, 0.0302, 0.0312, 0.0299, 0.04, 0.0799, 0.0278, 0.0269, 0.0393, 0.0263, 0.0378, 0.0339, 0.0308, 0.0306, 0.0282, 0.0323, 0.0276, 0.0272, 0.0338, 0.0394, 0.0298, 0.0301, 0.0311, 0.0406, 0.0319, 0.0317, 0.0281, 0.0277, 0.0769, 0.0547, 0.0385, 0.0273, 0.0278, 0.0317, 0.0323, 0.035, 0.0385, 0.0392, 0.0388, 0.0373, 0.0374, 0.0391, 0.0353, 0.0382, 0.0365, 0.0312, 0.0392, 0.0454, 0.0387, 0.0384, 0.0412, 0.0386, 0.0346, 0.0396, 0.0365, 0.0373, 0.0326, 0.0465, 0.0375, 0.0384, 0.0459, 0.0441, 0.0438, 0.0352, 0.0405, 0.0322, 0.0807, 0.0863, 0.0391, 0.0317, 0.0354, 0.0369, 0.0339, 0.0381, 0.0377, 0.0323, 0.0412, 0.0353, 0.0475, 0.0355, 0.0385, 0.0322, 0.0397, 0.0419, 0.0313, 0.0351, 0.042, 0.0584, 0.0428, 0.0428, 0.0744, 0.0314, 0.0448, 0.0363, 0.0304, 0.0325, 0.0314, 0.0493, 0.0377, 0.0313, 0.0336, 0.0343, 0.0317, 0.0342, 0.0342, 0.0333, 0.0369, 0.0341, 0.0328, 0.0451, 0.0294, 0.0311, 0.0324, 0.0324, 0.0343, 0.0326, 0.0375, 0.0346, 0.0438, 0.0433, 0.0381, 0.0329, 0.0387, 0.0399, 0.0621, 0.0312, 0.0439, 0.0302, 0.032, 0.0339, 0.0335, 0.0341, 0.03, 0.0302, 0.0368, 0.0335, 0.0379, 0.03, 0.0413, 0.0393, 0.0313, 0.0304, 0.0274, 0.029, 0.0336, 0.0309, 0.0341, 0.059, 0.0315, 0.0287, 0.0309, 0.031, 0.0307, 0.0275, 0.0293, 0.029, 0.0326, 0.048, 0.0316, 0.0298, 0.0862, 0.051, 0.0627, 0.0397, 0.0342, 0.0252, 0.0279, 0.0335, 0.0363, 0.0314, 0.0376, 0.0306, 0.0402, 0.0312, 0.0314, 0.0666, 0.0496, 0.0337, 0.032, 0.037, 0.0308, 0.0341, 0.0414, 0.0404, 0.0407, 0.0368, 0.0405, 0.0418, 0.0313, 0.0314, 0.0367, 0.0268, 0.0436, 0.0353, 0.0447, 0.0689, 0.0333, 0.0331, 0.0307, 0.0301, 0.0502, 0.0332, 0.0487, 0.0526, 0.0839, 0.0448, 0.0312, 0.0267, 0.035, 0.0303, 0.042, 0.0362, 0.0368, 0.0386, 0.033, 0.0388, 0.0446, 0.0368, 0.0456, 0.0322, 0.0304, 0.0269, 0.0264, 0.0793, 0.0287, 0.0323, 0.0346, 0.0353, 0.035, 0.0323, 0.0353, 0.0363, 0.0383, 0.0562]

    #### 204FR
    Spatial_204FR = [0.0257, 0.0703, 0.0287, 0.0574, 0.0634, 0.0472, 0.0242, 0.0252, 0.0262, 0.029, 0.0311, 0.0254, 0.0594, 0.0338, 0.0345, 0.0347, 0.0312, 0.0708, 0.0475, 0.0527, 0.0657, 0.0409, 0.0368, 0.0285, 0.0274, 0.0282, 0.0335, 0.0362, 0.0255, 0.0305, 0.0715, 0.05, 0.0694, 0.0539, 0.0314, 0.0295, 0.0306, 0.0305, 0.0321, 0.0324, 0.0286, 0.0474, 0.0318, 0.028, 0.0299, 0.0411, 0.0271, 0.0356, 0.0337, 0.0369, 0.0364, 0.0338, 0.0383, 0.0372, 0.0422, 0.0293, 0.0342, 0.0317, 0.0322, 0.0778, 0.0371, 0.0357, 0.0359, 0.0394, 0.0386, 0.0368, 0.034, 0.0307, 0.0424, 0.0386, 0.0367, 0.0386, 0.0393, 0.0498, 0.0374, 0.0382, 0.0831, 0.0696, 0.0575, 0.0333, 0.0333, 0.0372, 0.0376, 0.0371, 0.0552, 0.0354, 0.0347, 0.0365, 0.0402, 0.0402, 0.0404, 0.0431, 0.0353, 0.0354, 0.0329, 0.0666, 0.0421, 0.0293, 0.0316, 0.0373, 0.0496, 0.0465, 0.0527, 0.0486, 0.0479, 0.0523, 0.0485, 0.0483, 0.0546, 0.0478, 0.0458, 0.0532, 0.0515, 0.0496, 0.0496, 0.0483, 0.0504, 0.0485, 0.0421, 0.0433, 0.0513, 0.0466, 0.0516, 0.0506, 0.0472, 0.0506, 0.05, 0.0501, 0.0446, 0.0539, 0.0401, 0.0376, 0.0414, 0.0399, 0.0393, 0.0368, 0.0363, 0.0361, 0.0316, 0.0346, 0.0333, 0.0312, 0.0308, 0.0441, 0.0488, 0.0563, 0.0391, 0.0357, 0.0387, 0.0367, 0.0398, 0.0332, 0.0369, 0.0381, 0.0856, 0.0705, 0.0322, 0.0699, 0.0367, 0.035, 0.034, 0.0352, 0.042, 0.038, 0.0402, 0.0352, 0.0375, 0.0389, 0.0377, 0.0403, 0.0379, 0.0434, 0.0438, 0.0345, 0.0347, 0.0334, 0.0414, 0.035, 0.0584, 0.0665, 0.0435, 0.0377, 0.0365, 0.0356, 0.0426, 0.0415, 0.0403, 0.0426, 0.0408, 0.0391, 0.0394, 0.0421, 0.0379, 0.04, 0.036, 0.0433, 0.0369, 0.0337, 0.0362, 0.0319, 0.0375, 0.0325, 0.0445, 0.0363, 0.0354, 0.0296, 0.0377, 0.0407, 0.0642, 0.0434, 0.045, 0.0435, 0.0387, 0.0317, 0.0486, 0.0341, 0.045, 0.0554, 0.0306, 0.036, 0.0365, 0.0384, 0.0392, 0.0437, 0.0436, 0.0407, 0.0378, 0.0344, 0.04, 0.0416, 0.0433, 0.0414, 0.04, 0.0393, 0.0353, 0.0412, 0.0466, 0.0505, 0.0363, 0.0425, 0.0451, 0.0356, 0.0336, 0.037, 0.0414, 0.0394, 0.035, 0.0361, 0.0502, 0.0506, 0.0824, 0.0673, 0.035, 0.0314, 0.0336, 0.0345, 0.0364, 0.0447, 0.0403, 0.0435, 0.0402, 0.0526, 0.0577, 0.0332, 0.0492, 0.0395, 0.038, 0.0373, 0.0442, 0.0443, 0.0532, 0.0327, 0.0514, 0.05, 0.0469, 0.0356, 0.0308, 0.0301, 0.0299, 0.0374, 0.0592, 0.0598, 0.0495, 0.046, 0.0367, 0.0371, 0.0381, 0.0388]

    #### 206FRL
    Spatial_206FRL = [0.0333, 0.0329, 0.0401, 0.0351, 0.0404, 0.0294, 0.0349, 0.0456, 0.0298, 0.0328, 0.0309, 0.0317, 0.0309, 0.0319, 0.0415, 0.0309, 0.032, 0.0319, 0.0279, 0.0327, 0.0322, 0.0329, 0.0378, 0.0321, 0.0275, 0.0319, 0.0317, 0.0302, 0.0299, 0.0571, 0.0463, 0.0287, 0.0305, 0.0381, 0.0382, 0.0304, 0.0287, 0.0286, 0.0301, 0.0398, 0.0318, 0.0268, 0.0388, 0.0347, 0.028, 0.0312, 0.0277, 0.0309, 0.0245, 0.0274, 0.0301, 0.0279, 0.0275, 0.0482, 0.0515, 0.0299, 0.0619, 0.0503, 0.026, 0.029, 0.0352, 0.0349, 0.0643, 0.04, 0.0313, 0.0295, 0.0301, 0.0337, 0.0298, 0.0297, 0.0315, 0.0332, 0.0301, 0.0309, 0.0318, 0.0465, 0.0312, 0.0365, 0.0287, 0.0333, 0.0304, 0.0307, 0.0309, 0.0347, 0.0411, 0.0291, 0.0308, 0.0701, 0.0595, 0.0625, 0.0538, 0.0586, 0.0293, 0.0343, 0.0308, 0.0339, 0.0336, 0.0286, 0.0275, 0.0355, 0.0338, 0.0295, 0.038, 0.031, 0.036, 0.0533, 0.0343, 0.0333, 0.031, 0.0332, 0.0324, 0.0317, 0.0278, 0.0339, 0.0644, 0.0494, 0.037, 0.0393, 0.0369, 0.0312, 0.0411, 0.0391, 0.0389, 0.0382, 0.041, 0.0405, 0.0371, 0.0477, 0.0432, 0.043, 0.037, 0.0428, 0.0366, 0.0382, 0.0436, 0.0404, 0.0399, 0.0587, 0.0345, 0.0373, 0.036, 0.0373, 0.0368, 0.0378, 0.0392, 0.0389, 0.0402, 0.0382, 0.0379, 0.0376, 0.0374, 0.0358, 0.037, 0.0374, 0.0363, 0.0379, 0.0311, 0.0365, 0.034, 0.0342, 0.0326, 0.0373, 0.0334, 0.0327, 0.0328, 0.0375, 0.0332, 0.0342, 0.0372, 0.0467, 0.0286, 0.0304, 0.0315, 0.0341, 0.0348, 0.0324, 0.0337, 0.0347, 0.0343, 0.0345, 0.0328, 0.0326, 0.0338, 0.0334, 0.0335, 0.033, 0.0372, 0.0317, 0.0327, 0.0336, 0.0547, 0.043, 0.0429, 0.0308, 0.0327, 0.0331, 0.0353, 0.0773, 0.047, 0.0484, 0.0393, 0.0325, 0.0314, 0.0338, 0.034, 0.036, 0.0385, 0.0325, 0.0281, 0.0362, 0.0498, 0.0319, 0.0334, 0.0388, 0.0383, 0.0364, 0.037, 0.0386, 0.0486, 0.0337, 0.0364, 0.0365, 0.0379, 0.0365, 0.0381, 0.0408, 0.0391, 0.0385, 0.0365, 0.0358, 0.0432, 0.0423, 0.0425, 0.0477, 0.0533, 0.0315, 0.0334, 0.037, 0.0394, 0.0341, 0.0366, 0.0402, 0.0338, 0.0362, 0.0366, 0.032, 0.035, 0.0343, 0.0329, 0.0334, 0.0344, 0.0316, 0.0337, 0.0334, 0.0772, 0.0648, 0.0481, 0.0377, 0.0337, 0.0365, 0.0341, 0.0287, 0.0434, 0.0401, 0.0372, 0.0375, 0.0347, 0.0601, 0.0324, 0.0335, 0.0399, 0.0422, 0.0415, 0.0417, 0.0405, 0.0382, 0.0404, 0.0402, 0.035, 0.0355, 0.0349, 0.0412, 0.0351, 0.0432, 0.0369, 0.0303, 0.0335, 0.0343, 0.0399, 0.0367, 0.0393, 0.0343, 0.0321, 0.0579, 0.0603, 0.0327, 0.0323, 0.0348, 0.033, 0.0342, 0.0319, 0.0303, 0.0341, 0.0328, 0.0441, 0.0491, 0.0412, 0.0406, 0.0386, 0.0369, 0.0351, 0.041, 0.0414, 0.0376, 0.037, 0.0358, 0.0357, 0.0424, 0.043, 0.0485, 0.0507, 0.0525, 0.0524, 0.062, 0.0585, 0.0524, 0.0509, 0.057, 0.0558, 0.062, 0.0512]

    #### 211MRR
    Spatial_211MRR = [0.0369, 0.0359, 0.0395, 0.0353, 0.037, 0.0328, 0.0396, 0.0405, 0.0344, 0.0435, 0.0391, 0.0386, 0.0397, 0.0535, 0.0322, 0.0351, 0.0406, 0.0421, 0.0374, 0.0419, 0.0489, 0.041, 0.0452, 0.0534, 0.053, 0.0428, 0.0516, 0.043, 0.0456, 0.0416, 0.0288, 0.038, 0.0353, 0.0334, 0.0326, 0.0347, 0.0366, 0.0401, 0.0437, 0.0467, 0.0551, 0.0422, 0.0447, 0.0363, 0.037, 0.0347, 0.0342, 0.0418, 0.0452, 0.0489, 0.0363, 0.0417, 0.0355, 0.045, 0.0299, 0.0421, 0.0348, 0.0339, 0.0339, 0.032, 0.042, 0.0438, 0.0568, 0.037, 0.0467, 0.0472, 0.044, 0.0498, 0.039, 0.0384, 0.0377, 0.0394, 0.0443, 0.0389, 0.0351, 0.047, 0.0466, 0.0402, 0.0453, 0.0412, 0.0501, 0.0489, 0.0465, 0.0344, 0.0512, 0.0432, 0.0287, 0.0251, 0.0305, 0.0365, 0.0396, 0.0484, 0.0281, 0.0255, 0.0285, 0.0368, 0.0295, 0.0299, 0.0288, 0.0329, 0.035, 0.0289, 0.0341, 0.0286, 0.0311, 0.0363, 0.0334, 0.0318, 0.0257, 0.0293, 0.0267, 0.0303, 0.0321, 0.028, 0.0451, 0.0682, 0.046, 0.05, 0.0522, 0.05, 0.0311, 0.0483, 0.0447, 0.0459, 0.0359, 0.0502, 0.0333, 0.0388, 0.0386, 0.0379, 0.0455, 0.0385, 0.0604, 0.0565, 0.0337, 0.033, 0.0352, 0.0451, 0.0458, 0.0303, 0.0445, 0.06, 0.045, 0.0274, 0.0373, 0.0328, 0.0394, 0.0411, 0.0436, 0.039, 0.0356, 0.0357, 0.0277, 0.0305, 0.0406, 0.0335, 0.0251, 0.0292, 0.0254, 0.0276, 0.0254, 0.0336, 0.0404, 0.0285, 0.0253, 0.0214, 0.0244, 0.0253, 0.0278, 0.0276, 0.0309, 0.0312, 0.031, 0.0252, 0.0336, 0.0315, 0.029, 0.0222, 0.0347, 0.0588, 0.022, 0.05, 0.0239, 0.0394, 0.0346, 0.0297, 0.0357, 0.0331, 0.0348, 0.0291, 0.0315, 0.0677, 0.0321, 0.0273, 0.0371, 0.031, 0.0352, 0.0309, 0.0335, 0.0308, 0.0319, 0.0396, 0.0412, 0.0403, 0.0418, 0.0314, 0.0272, 0.0526, 0.0314, 0.0372, 0.0401, 0.0433, 0.0478, 0.0399, 0.0415, 0.0328, 0.0351, 0.0445, 0.0461, 0.0512, 0.0371, 0.0335, 0.0358, 0.0298, 0.0402, 0.0636, 0.0381, 0.0424, 0.0353, 0.0335, 0.0314, 0.0464, 0.0365, 0.0365, 0.057, 0.043, 0.034, 0.0543, 0.0442, 0.0383, 0.0296, 0.0341, 0.044, 0.0472, 0.0287, 0.037, 0.0374, 0.0374, 0.039, 0.0481, 0.0392, 0.055, 0.0556, 0.0538, 0.0379, 0.0551, 0.05, 0.0454, 0.0415, 0.0384, 0.0429, 0.0479, 0.0455, 0.0679, 0.0323, 0.0298, 0.0366, 0.0384, 0.0321, 0.0326, 0.0355, 0.0385, 0.0453, 0.0422, 0.0449, 0.0477, 0.0445, 0.0352, 0.0403, 0.0355, 0.0481, 0.0456, 0.0331, 0.0492, 0.0514, 0.0325, 0.0347, 0.035, 0.03, 0.0425, 0.0391, 0.04, 0.0488, 0.0518, 0.0404, 0.0369]

    #### 218MN
    Spatial_218MN = [0.0649, 0.0285, 0.0235, 0.065, 0.0267, 0.0439, 0.0266, 0.0269, 0.0461, 0.0336, 0.0311, 0.0273, 0.0316, 0.0746, 0.0418, 0.0337, 0.028, 0.0575, 0.0338, 0.0287, 0.0293, 0.0261, 0.0263, 0.0551, 0.0315, 0.0335, 0.0284, 0.0292, 0.03, 0.0306, 0.0288, 0.0298, 0.0309, 0.0286, 0.0319, 0.0533, 0.0271, 0.0286, 0.0264, 0.0303, 0.0296, 0.0295, 0.028, 0.0448, 0.0318, 0.0321, 0.0291, 0.0356, 0.033, 0.0316, 0.0305, 0.0315, 0.0697, 0.0381, 0.0325, 0.032, 0.0272, 0.0311, 0.0305, 0.0302, 0.0338, 0.0341, 0.0304, 0.0257, 0.0305, 0.032, 0.0325, 0.0328, 0.0336, 0.0302, 0.0398, 0.0367, 0.0688, 0.0305, 0.0357, 0.0368, 0.0323, 0.0417, 0.0328, 0.0313, 0.0308, 0.0263, 0.038, 0.0656, 0.0366, 0.0314, 0.0285, 0.0395, 0.0422, 0.0394, 0.0392, 0.0408, 0.0398, 0.0495, 0.0442, 0.0382, 0.04, 0.0492, 0.0418, 0.0646, 0.0337, 0.0312, 0.031, 0.0351, 0.032, 0.0316, 0.0337, 0.0318, 0.0442, 0.0388, 0.0417, 0.0424, 0.0465, 0.0483, 0.046, 0.0383, 0.0425, 0.0427, 0.0475, 0.0491, 0.0309, 0.0271, 0.0298, 0.0275, 0.0276, 0.0364, 0.0325, 0.0313, 0.0349, 0.0256, 0.0261, 0.0312, 0.0262, 0.0261, 0.0259, 0.0264, 0.0283, 0.0281, 0.0298, 0.028, 0.0767, 0.0294, 0.0275, 0.0288, 0.0338, 0.0268, 0.0371, 0.0461, 0.0429, 0.0337, 0.0465, 0.0467, 0.0554, 0.0576, 0.0242, 0.052, 0.0299, 0.0319, 0.0331, 0.0317, 0.0312, 0.0328, 0.0352, 0.0356, 0.0365, 0.0399, 0.0483, 0.0501, 0.0366, 0.0404, 0.0453, 0.0323, 0.0333, 0.0424, 0.0323, 0.0357, 0.0442, 0.0373, 0.0353, 0.0315, 0.0427, 0.0509, 0.0431, 0.0468, 0.0569, 0.0706, 0.0382, 0.0384, 0.0418, 0.0612, 0.0542, 0.0325, 0.0494, 0.0462, 0.0333, 0.0405, 0.0304, 0.0316, 0.035, 0.0317, 0.047, 0.0528, 0.0473, 0.0358, 0.0351, 0.0395, 0.0365, 0.0357, 0.0278, 0.0386, 0.0298, 0.0331, 0.0446, 0.0409, 0.048, 0.0429, 0.0425, 0.0459, 0.0304, 0.031, 0.0361, 0.0444, 0.0468, 0.0509, 0.0429, 0.0428, 0.0308, 0.0305, 0.0679, 0.0388, 0.0428, 0.0444, 0.0523, 0.0506, 0.0436, 0.0509, 0.0505, 0.0486, 0.0424, 0.051, 0.0433, 0.0444, 0.0448, 0.0352, 0.0475, 0.0318, 0.032, 0.0354, 0.0518, 0.0466, 0.031, 0.0346, 0.0337, 0.0415, 0.0346, 0.0293, 0.0463, 0.0355, 0.0343, 0.0481, 0.0382]

    #### 21ML
    Spatial_21ML = [0.0262, 0.0568, 0.0309, 0.0369, 0.0446, 0.0667, 0.0355, 0.0736, 0.0267, 0.0436, 0.031, 0.0308, 0.031, 0.0701, 0.0392, 0.0233, 0.0307, 0.0436, 0.0246, 0.0722, 0.0705, 0.0346, 0.0235, 0.0365, 0.0314, 0.0242, 0.0532, 0.0231, 0.0354, 0.0395, 0.0378, 0.0218, 0.0689, 0.0386, 0.0536, 0.0726, 0.0529, 0.0292, 0.0721, 0.0689, 0.052, 0.0388, 0.0287, 0.0194, 0.0323, 0.0268, 0.0395, 0.0228, 0.0273, 0.0409, 0.0318, 0.0263]

    Spatial = Spatial_206FRL + Spatial_204FR + Spatial_203MN + Spatial_187FN + Spatial_211MRR + Spatial_63MR + Spatial_218MN + Spatial_54MRL + Spatial_21ML
    return Spatial

def data_sigma_5():
    #### 54MRL
    Spatial_54MRL = [0.0322, 0.0344, 0.0336, 0.0646, 0.0534, 0.0552, 0.0301, 0.0478, 0.0505, 0.0731, 0.0676, 0.0464, 0.0504, 0.0575, 0.0395, 0.0597, 0.0602, 0.0313, 0.0563, 0.0459, 0.0391, 0.0323, 0.0463, 0.0397, 0.0289, 0.0382, 0.0814, 0.0664, 0.0451, 0.0349, 0.0474, 0.0773, 0.0407, 0.0477, 0.0314, 0.043, 0.0775, 0.0444, 0.0404, 0.0545, 0.036, 0.0512, 0.048, 0.0729, 0.0417, 0.0368, 0.033, 0.0428, 0.0332, 0.0592, 0.0487, 0.0947, 0.0376, 0.0516, 0.0565, 0.0318, 0.0341, 0.0393, 0.0541, 0.0506, 0.0351, 0.0374, 0.0543, 0.0535, 0.0603, 0.0377, 0.0476, 0.0473, 0.0435, 0.0468, 0.0512, 0.0388, 0.0685, 0.0673, 0.0519, 0.0654, 0.0558, 0.0586, 0.0659, 0.0427, 0.0457, 0.0392, 0.0604, 0.0444, 0.056, 0.0458, 0.0754, 0.0466, 0.0376, 0.0474, 0.0544, 0.0449, 0.0292, 0.0591, 0.0469, 0.0758, 0.067, 0.0519, 0.0382, 0.0415, 0.0689, 0.0382, 0.0477, 0.03, 0.065, 0.0455, 0.0512, 0.039, 0.05, 0.0331, 0.0329, 0.0364, 0.0617, 0.0351, 0.0843, 0.0375, 0.0331, 0.0341, 0.0515, 0.0397, 0.034, 0.0329, 0.0303, 0.07, 0.0392, 0.0382, 0.0475, 0.0311, 0.0582, 0.0347, 0.0583, 0.038, 0.0344, 0.036, 0.0636, 0.0685, 0.0375, 0.0309, 0.0429, 0.0722, 0.052, 0.038, 0.0368, 0.0617, 0.0623, 0.0308, 0.0501, 0.0456, 0.0516, 0.0662, 0.034, 0.0354, 0.0442, 0.0347, 0.0577, 0.0296, 0.0357, 0.0515, 0.0377, 0.0295, 0.0424, 0.0618, 0.0622, 0.0265, 0.0365, 0.037, 0.037, 0.0647, 0.0758, 0.0414, 0.0503, 0.0833, 0.0404, 0.0411, 0.0326, 0.0575, 0.047, 0.0744, 0.0445, 0.0476, 0.031, 0.0311, 0.0722, 0.0356, 0.0348, 0.0322, 0.0341, 0.0312, 0.0846, 0.0636, 0.0558, 0.039, 0.0353, 0.0362, 0.0483, 0.0538, 0.0438, 0.0539, 0.0344, 0.0718, 0.0526, 0.0593, 0.0789, 0.0516, 0.0304, 0.0358, 0.0494, 0.03, 0.0783, 0.0431, 0.0298, 0.036, 0.0401, 0.0333, 0.0864, 0.0358, 0.031, 0.0292, 0.0286, 0.0292, 0.0547, 0.0497, 0.0359, 0.0673, 0.0404, 0.0486, 0.0495, 0.0744, 0.0579, 0.0684, 0.0815, 0.0348, 0.0791, 0.0384, 0.0436, 0.0361, 0.08, 0.0339, 0.0346, 0.033, 0.037, 0.0482, 0.0388, 0.0441, 0.0428, 0.0447, 0.0438, 0.0505, 0.0434, 0.0448, 0.0657, 0.0661, 0.0379, 0.0344, 0.0586, 0.0457, 0.0631, 0.0425, 0.0493, 0.0505, 0.0519, 0.0404, 0.0542, 0.0365, 0.037, 0.0427, 0.0599, 0.0533, 0.035, 0.0396, 0.0395, 0.0476, 0.0393, 0.0331, 0.0524, 0.0409, 0.0393, 0.0538, 0.0427]

    #### 63MR
    Spatial_63MR = [0.0446, 0.0269, 0.0263, 0.0294, 0.0432, 0.0572, 0.041, 0.0246, 0.0414, 0.0417, 0.027, 0.0339, 0.0413, 0.0288, 0.0461, 0.0373, 0.0645, 0.0297, 0.029, 0.0662, 0.0339, 0.0328, 0.0308, 0.0455, 0.0278, 0.0424, 0.0287, 0.0273, 0.0562, 0.0576, 0.0267, 0.0316, 0.0359, 0.0381, 0.0482, 0.0351, 0.0669, 0.0327, 0.0454, 0.0594, 0.0439, 0.037, 0.0341, 0.0249, 0.0322, 0.0298, 0.0309, 0.0365, 0.0476, 0.0358, 0.0286, 0.0287, 0.0387, 0.0571, 0.0559, 0.0403, 0.0449, 0.029, 0.0671, 0.0335, 0.0484, 0.04, 0.0353, 0.0609, 0.0814, 0.0321, 0.0466, 0.0318, 0.0342, 0.0425, 0.0447, 0.042, 0.0351, 0.0587, 0.0586, 0.0324, 0.067, 0.0398, 0.0373, 0.0435, 0.0581, 0.044, 0.0413, 0.0321, 0.0267, 0.0249, 0.0265, 0.0537, 0.0293, 0.0268, 0.0307, 0.0782, 0.0633, 0.0371, 0.0523, 0.0264, 0.06, 0.0365, 0.0475, 0.0805, 0.0422, 0.0549, 0.0449, 0.0648, 0.0449, 0.0513, 0.0552, 0.0628, 0.0431, 0.0451, 0.052, 0.0265, 0.0308, 0.0377, 0.0365, 0.027, 0.0373, 0.0272, 0.0407, 0.0504, 0.0553, 0.0466, 0.031, 0.0567, 0.079, 0.0958, 0.024, 0.0337, 0.0311, 0.027, 0.0319, 0.0383, 0.0514, 0.0688, 0.0611, 0.0355, 0.0268, 0.0587, 0.0238, 0.0733, 0.0474, 0.0381, 0.07, 0.0827, 0.0398, 0.0437, 0.028, 0.0785, 0.0378, 0.0603, 0.037, 0.0494, 0.0622, 0.0315, 0.0394, 0.0312, 0.0515, 0.0627, 0.0564, 0.0605, 0.0526, 0.0294, 0.0324, 0.0278, 0.0349, 0.0423, 0.0319, 0.0795, 0.0388, 0.0288, 0.0636, 0.0303, 0.0372, 0.0345, 0.0403, 0.0507, 0.0313, 0.038, 0.0972, 0.0303, 0.0321, 0.0389, 0.0361, 0.0338, 0.0461, 0.0295, 0.0741, 0.0455, 0.0365, 0.0383, 0.0304, 0.03, 0.0323, 0.0331, 0.0356, 0.0481, 0.0673, 0.0348, 0.0737, 0.0416, 0.0501, 0.0196, 0.0645, 0.0551, 0.0558, 0.0825, 0.0352, 0.0239, 0.0574, 0.0611, 0.0454]

    #### 187FN
    Spatial_187FN = [0.04, 0.0335, 0.0397, 0.0807, 0.0544, 0.0777, 0.0362, 0.0315, 0.03, 0.0355, 0.031, 0.0331, 0.0314, 0.0358, 0.0365, 0.0377, 0.032, 0.0397, 0.0357, 0.037, 0.0345, 0.038, 0.0332, 0.0457, 0.0343, 0.0371, 0.0331, 0.036, 0.0589, 0.0325, 0.0311, 0.0354, 0.0417, 0.0287, 0.0329, 0.0356, 0.034, 0.0364, 0.0313, 0.0336, 0.0389, 0.0304, 0.0321, 0.0311, 0.0488, 0.0576, 0.0473, 0.0682, 0.0544, 0.069, 0.0479, 0.0375, 0.0618, 0.0483, 0.0429, 0.0625, 0.0421, 0.0503, 0.0533, 0.0484, 0.0389, 0.0439, 0.0358, 0.0498, 0.0412, 0.0418, 0.0426, 0.0517, 0.0404, 0.038, 0.0568, 0.054, 0.0417, 0.0523, 0.0498, 0.059, 0.057, 0.0517, 0.0485, 0.0472, 0.0573, 0.0462, 0.0382, 0.0468, 0.0487, 0.0368, 0.0605, 0.052, 0.0617, 0.0898, 0.0475, 0.0501, 0.0469, 0.0418, 0.0729, 0.0567, 0.0581, 0.061, 0.073, 0.0565, 0.0713, 0.0376, 0.045, 0.0365, 0.0389, 0.0492, 0.0523, 0.0325, 0.0298, 0.0341, 0.0376, 0.0348, 0.0444, 0.0367, 0.0367, 0.0411, 0.0407, 0.0586, 0.0278, 0.0358, 0.0447, 0.0484, 0.034, 0.0493, 0.041, 0.0536, 0.053, 0.0379, 0.0533, 0.1022, 0.0364, 0.0345, 0.035, 0.0295, 0.026, 0.0465, 0.0433, 0.0516, 0.0404, 0.0617, 0.0497, 0.0537, 0.0359, 0.0426, 0.0553, 0.046, 0.0517, 0.0577, 0.0571, 0.0456, 0.0359, 0.038, 0.0373, 0.0507, 0.0436, 0.0431, 0.0571, 0.0531, 0.0493, 0.037, 0.0435, 0.0329, 0.0479, 0.0408, 0.0513, 0.0478, 0.0548, 0.0594, 0.0457, 0.0611, 0.0593, 0.0362, 0.0709, 0.0325, 0.0486, 0.0419, 0.0416, 0.0399, 0.0449, 0.067, 0.0691, 0.0575, 0.0517, 0.0622, 0.064, 0.0472, 0.0688, 0.0426, 0.0614, 0.0642, 0.034, 0.0384, 0.0471, 0.0349, 0.0753, 0.0477, 0.072, 0.0434, 0.0375, 0.0407, 0.0499, 0.0386, 0.0364, 0.0496, 0.0958, 0.0444, 0.0429, 0.0489, 0.0666, 0.0422, 0.0442, 0.0564, 0.0316, 0.0444, 0.0447, 0.0503, 0.0351, 0.0415, 0.0393, 0.0383, 0.0449, 0.0462, 0.0379, 0.0377, 0.0628, 0.0466, 0.0538, 0.0482, 0.037, 0.0475, 0.0491, 0.0543, 0.06, 0.0562, 0.061, 0.0349, 0.0432, 0.0399, 0.0558, 0.0378, 0.0372, 0.0352, 0.0425, 0.0352, 0.0352, 0.0438, 0.0582, 0.0516, 0.0612, 0.054, 0.0474, 0.0346, 0.0354, 0.0961, 0.0365, 0.0348, 0.0405, 0.0467, 0.044, 0.0362, 0.0374, 0.0287, 0.0434, 0.0559, 0.0356, 0.0569, 0.0482, 0.0422, 0.0542, 0.0509, 0.0505, 0.0828, 0.0695, 0.0474, 0.0423, 0.0473, 0.0386, 0.0408, 0.057, 0.0382, 0.0352, 0.0471, 0.0399, 0.0476, 0.0535, 0.0546, 0.0507, 0.0451, 0.0396, 0.0411, 0.0474, 0.048, 0.0555, 0.0851, 0.0348, 0.0406, 0.0352, 0.0628, 0.0433, 0.0403, 0.0508, 0.0491, 0.0566, 0.0504, 0.0686, 0.0576, 0.0544, 0.0451, 0.0655, 0.0567, 0.0511, 0.0789, 0.0625, 0.0586, 0.0363, 0.0435, 0.0693, 0.0897, 0.0471, 0.037, 0.0302, 0.0588, 0.0351, 0.0335, 0.038, 0.0401, 0.0366, 0.0337, 0.0484, 0.0403, 0.0541, 0.0296, 0.0376, 0.0363, 0.042, 0.0757, 0.0376, 0.0431, 0.0784, 0.053, 0.034, 0.0471, 0.045, 0.0451, 0.048, 0.0545, 0.0618, 0.0706, 0.0446, 0.0771, 0.0456, 0.0886, 0.0308, 0.0402, 0.0346, 0.0381, 0.0365, 0.0326, 0.0365, 0.0358, 0.0408, 0.0904, 0.0768, 0.0551, 0.0635, 0.0318, 0.0266, 0.0444, 0.0359, 0.0317, 0.0368, 0.0302, 0.066, 0.0319, 0.031, 0.0951, 0.0419, 0.0326, 0.0415, 0.0292, 0.0329, 0.0352, 0.046, 0.0343, 0.0353]

    #### 203MN
    Spatial_203MN = [0.032, 0.0363, 0.041, 0.0302, 0.0305, 0.0321, 0.0327, 0.032, 0.0414, 0.0337, 0.0305, 0.0409, 0.0433, 0.0319, 0.0737, 0.0414, 0.0773, 0.0358, 0.0332, 0.0332, 0.0398, 0.0396, 0.0336, 0.0309, 0.0467, 0.0367, 0.037, 0.0341, 0.0363, 0.0341, 0.0352, 0.0368, 0.0349, 0.0357, 0.0551, 0.0411, 0.0415, 0.0451, 0.0503, 0.0391, 0.0446, 0.0489, 0.0419, 0.0434, 0.0434, 0.0502, 0.0439, 0.046, 0.0498, 0.0499, 0.0452, 0.0413, 0.0497, 0.0469, 0.0473, 0.0407, 0.0379, 0.0386, 0.0482, 0.0469, 0.0443, 0.0446, 0.0424, 0.0862, 0.0509, 0.0474, 0.0384, 0.0419, 0.0424, 0.0491, 0.0484, 0.0465, 0.0454, 0.0445, 0.0483, 0.0502, 0.0454, 0.041, 0.0564, 0.0424, 0.0456, 0.0435, 0.0389, 0.0686, 0.0386, 0.0451, 0.0424, 0.0422, 0.0434, 0.0543, 0.0319, 0.0307, 0.0339, 0.0341, 0.0344, 0.0476, 0.0368, 0.0389, 0.0329, 0.0336, 0.0347, 0.0335, 0.0446, 0.089, 0.0311, 0.0297, 0.0444, 0.0296, 0.0441, 0.039, 0.035, 0.034, 0.0315, 0.0358, 0.0307, 0.0302, 0.0381, 0.0439, 0.0331, 0.0339, 0.0349, 0.0459, 0.0361, 0.0357, 0.0317, 0.0308, 0.0867, 0.0601, 0.043, 0.0301, 0.0307, 0.0355, 0.0361, 0.0398, 0.0441, 0.0447, 0.0441, 0.0422, 0.0423, 0.044, 0.0399, 0.0427, 0.0408, 0.0346, 0.0445, 0.0524, 0.0438, 0.0428, 0.046, 0.0432, 0.0388, 0.0448, 0.041, 0.042, 0.0361, 0.0514, 0.0423, 0.0442, 0.0525, 0.0497, 0.0497, 0.0391, 0.0448, 0.0358, 0.0886, 0.0946, 0.0437, 0.035, 0.0396, 0.0419, 0.0385, 0.0429, 0.0426, 0.0363, 0.0472, 0.0398, 0.054, 0.0403, 0.044, 0.0359, 0.0448, 0.0475, 0.0351, 0.0399, 0.0483, 0.067, 0.0491, 0.049, 0.085, 0.0352, 0.0497, 0.0405, 0.0341, 0.0368, 0.0352, 0.0554, 0.042, 0.0345, 0.038, 0.0387, 0.0355, 0.0386, 0.0385, 0.0374, 0.0417, 0.0382, 0.0366, 0.0506, 0.0328, 0.0342, 0.0361, 0.0362, 0.0383, 0.0364, 0.0424, 0.0392, 0.05, 0.0494, 0.043, 0.037, 0.0433, 0.0449, 0.0694, 0.0341, 0.0498, 0.0337, 0.0362, 0.0384, 0.0374, 0.0386, 0.0339, 0.0341, 0.0415, 0.0377, 0.0428, 0.0337, 0.0458, 0.0435, 0.0348, 0.0338, 0.0306, 0.0326, 0.0381, 0.0349, 0.0387, 0.065, 0.0345, 0.0321, 0.0353, 0.035, 0.0349, 0.0306, 0.0329, 0.0326, 0.0367, 0.0538, 0.0356, 0.0331, 0.0968, 0.057, 0.0688, 0.0444, 0.0384, 0.0279, 0.031, 0.0378, 0.0406, 0.0358, 0.0431, 0.0348, 0.0456, 0.0354, 0.036, 0.0753, 0.0552, 0.0383, 0.0361, 0.0418, 0.0347, 0.0386, 0.0473, 0.046, 0.0462, 0.0415, 0.0462, 0.0477, 0.0352, 0.0355, 0.0413, 0.03, 0.0497, 0.04, 0.0508, 0.0764, 0.0375, 0.0373, 0.0348, 0.0338, 0.0556, 0.037, 0.0544, 0.0586, 0.0922, 0.0497, 0.0355, 0.0297, 0.0399, 0.0348, 0.0489, 0.0411, 0.042, 0.0444, 0.0382, 0.0449, 0.0514, 0.0418, 0.0521, 0.0363, 0.0336, 0.0297, 0.0292, 0.0882, 0.0324, 0.0366, 0.0394, 0.0399, 0.0395, 0.0366, 0.0401, 0.041, 0.0431, 0.0616]

    #### 204FR
    Spatial_204FR = [0.0285, 0.0783, 0.032, 0.0652, 0.0696, 0.0536, 0.0271, 0.0282, 0.0296, 0.0329, 0.0352, 0.0285, 0.0649, 0.0376, 0.0383, 0.0394, 0.0351, 0.0787, 0.0514, 0.057, 0.0731, 0.0453, 0.0411, 0.0318, 0.0307, 0.0317, 0.0375, 0.041, 0.0282, 0.0339, 0.0789, 0.0528, 0.0753, 0.0591, 0.0351, 0.033, 0.0345, 0.0338, 0.0357, 0.0361, 0.0315, 0.0529, 0.0355, 0.031, 0.0335, 0.0461, 0.0301, 0.0402, 0.0378, 0.0409, 0.0405, 0.0375, 0.0424, 0.0413, 0.0468, 0.0323, 0.0383, 0.0353, 0.036, 0.0865, 0.0407, 0.0395, 0.0402, 0.044, 0.043, 0.0412, 0.0386, 0.0347, 0.0473, 0.0429, 0.041, 0.0433, 0.044, 0.0551, 0.0415, 0.0426, 0.0913, 0.0767, 0.0639, 0.0374, 0.0372, 0.0423, 0.043, 0.0418, 0.0612, 0.0399, 0.0392, 0.0417, 0.0455, 0.0456, 0.0461, 0.0478, 0.0394, 0.0397, 0.0368, 0.0733, 0.0468, 0.0325, 0.0353, 0.0423, 0.0556, 0.0519, 0.0593, 0.0542, 0.0537, 0.0592, 0.0546, 0.054, 0.0613, 0.0535, 0.0511, 0.0602, 0.0583, 0.056, 0.0556, 0.0548, 0.0567, 0.0545, 0.0463, 0.0481, 0.0574, 0.0517, 0.0577, 0.0575, 0.0523, 0.0567, 0.0559, 0.0561, 0.0499, 0.0616, 0.0449, 0.0422, 0.0464, 0.045, 0.0444, 0.0416, 0.0404, 0.0402, 0.0351, 0.0385, 0.0373, 0.0344, 0.0338, 0.0502, 0.0551, 0.0637, 0.0426, 0.0393, 0.0436, 0.0412, 0.0454, 0.037, 0.0415, 0.0427, 0.0943, 0.0777, 0.0355, 0.0773, 0.0412, 0.0393, 0.0381, 0.0396, 0.0476, 0.0422, 0.0456, 0.0398, 0.0427, 0.044, 0.0426, 0.0458, 0.0431, 0.0497, 0.0483, 0.0384, 0.0386, 0.0372, 0.0463, 0.0388, 0.0651, 0.0741, 0.0463, 0.0419, 0.0405, 0.04, 0.0479, 0.0472, 0.0459, 0.0482, 0.046, 0.044, 0.0444, 0.0476, 0.0426, 0.045, 0.0399, 0.0489, 0.0403, 0.0375, 0.0406, 0.0353, 0.042, 0.0361, 0.049, 0.0401, 0.0392, 0.0327, 0.0422, 0.0462, 0.0707, 0.0479, 0.0495, 0.0482, 0.0432, 0.0348, 0.0538, 0.0373, 0.0504, 0.0616, 0.0333, 0.0404, 0.0409, 0.0434, 0.0447, 0.0492, 0.0493, 0.0462, 0.0425, 0.0383, 0.0455, 0.0475, 0.049, 0.0468, 0.0451, 0.0433, 0.0387, 0.0462, 0.0532, 0.0577, 0.0408, 0.0492, 0.0501, 0.0391, 0.0369, 0.0411, 0.0466, 0.0448, 0.0388, 0.0401, 0.0561, 0.0562, 0.0884, 0.0723, 0.0382, 0.0342, 0.0377, 0.0386, 0.0408, 0.0508, 0.0454, 0.0483, 0.0449, 0.0578, 0.0635, 0.0359, 0.0545, 0.0437, 0.0422, 0.0415, 0.0508, 0.0505, 0.0615, 0.0367, 0.0574, 0.0553, 0.0515, 0.0392, 0.0344, 0.034, 0.0334, 0.0422, 0.0652, 0.066, 0.0564, 0.0518, 0.0413, 0.0422, 0.0434, 0.0446]

    #### 206FRL
    Spatial_206FRL = [0.038, 0.0376, 0.0462, 0.0399, 0.0459, 0.0334, 0.04, 0.0524, 0.0335, 0.0373, 0.0349, 0.0361, 0.035, 0.036, 0.048, 0.0353, 0.0363, 0.0362, 0.0311, 0.0372, 0.0366, 0.0373, 0.0431, 0.036, 0.0306, 0.0361, 0.0358, 0.0341, 0.034, 0.0646, 0.0517, 0.0323, 0.0342, 0.0433, 0.0433, 0.034, 0.032, 0.0318, 0.0343, 0.0452, 0.0358, 0.0298, 0.0435, 0.0392, 0.0315, 0.0354, 0.0312, 0.0342, 0.0275, 0.0307, 0.0339, 0.0312, 0.031, 0.0548, 0.0588, 0.0336, 0.0683, 0.0563, 0.0295, 0.0326, 0.0397, 0.0394, 0.072, 0.0457, 0.0355, 0.0335, 0.0342, 0.0375, 0.0335, 0.0339, 0.0356, 0.0373, 0.0344, 0.0344, 0.0361, 0.052, 0.0356, 0.0406, 0.0328, 0.0371, 0.0344, 0.0343, 0.0347, 0.0392, 0.0461, 0.0326, 0.0346, 0.0782, 0.0662, 0.0695, 0.06, 0.0652, 0.0333, 0.0389, 0.0355, 0.0383, 0.0376, 0.0323, 0.0313, 0.0397, 0.0382, 0.0338, 0.0431, 0.0357, 0.0411, 0.0596, 0.0389, 0.0379, 0.0356, 0.0383, 0.0371, 0.036, 0.0321, 0.0385, 0.0727, 0.0555, 0.0418, 0.0443, 0.0419, 0.0351, 0.0465, 0.044, 0.0444, 0.0432, 0.0474, 0.046, 0.0425, 0.0542, 0.049, 0.0491, 0.0418, 0.0487, 0.0413, 0.0432, 0.05, 0.0457, 0.0453, 0.066, 0.0386, 0.0421, 0.0405, 0.042, 0.0416, 0.0427, 0.0444, 0.0439, 0.0446, 0.0433, 0.0428, 0.0426, 0.0421, 0.04, 0.0417, 0.0425, 0.0412, 0.0423, 0.0343, 0.0407, 0.0382, 0.0377, 0.0366, 0.0422, 0.0376, 0.0364, 0.0368, 0.0423, 0.0377, 0.0387, 0.0416, 0.0523, 0.0317, 0.0336, 0.0352, 0.0385, 0.0392, 0.0368, 0.0381, 0.0394, 0.0391, 0.0389, 0.0368, 0.0365, 0.0378, 0.0375, 0.0371, 0.0373, 0.0422, 0.0356, 0.0366, 0.0378, 0.0613, 0.0476, 0.0482, 0.0344, 0.0369, 0.0371, 0.04, 0.0865, 0.0531, 0.0542, 0.0443, 0.0369, 0.035, 0.038, 0.0386, 0.0405, 0.0441, 0.0361, 0.0312, 0.0409, 0.0558, 0.0355, 0.0375, 0.0439, 0.0428, 0.0412, 0.0422, 0.0442, 0.0549, 0.0376, 0.0412, 0.041, 0.0431, 0.041, 0.0428, 0.0466, 0.0443, 0.0436, 0.0415, 0.0405, 0.0496, 0.0482, 0.0486, 0.0542, 0.0599, 0.0349, 0.038, 0.042, 0.0443, 0.0383, 0.0415, 0.0455, 0.0379, 0.0408, 0.0412, 0.0355, 0.039, 0.0382, 0.0366, 0.0374, 0.0383, 0.0353, 0.0377, 0.0374, 0.087, 0.0722, 0.0546, 0.0426, 0.0378, 0.0409, 0.0385, 0.032, 0.0498, 0.0457, 0.042, 0.0425, 0.0393, 0.0677, 0.0363, 0.0381, 0.046, 0.0478, 0.0474, 0.048, 0.0462, 0.0438, 0.0464, 0.045, 0.0393, 0.0403, 0.0395, 0.0473, 0.0391, 0.0491, 0.041, 0.0336, 0.0371, 0.0384, 0.0454, 0.0414, 0.0446, 0.0385, 0.0354, 0.066, 0.0665, 0.0364, 0.0361, 0.0393, 0.0369, 0.0386, 0.0359, 0.0333, 0.0381, 0.0365, 0.0502, 0.056, 0.0463, 0.0457, 0.0444, 0.0421, 0.0401, 0.0469, 0.0477, 0.043, 0.0419, 0.0406, 0.0405, 0.0467, 0.0475, 0.0541, 0.0579, 0.0599, 0.0602, 0.0707, 0.0665, 0.0593, 0.0573, 0.0651, 0.0636, 0.0707, 0.0578]

    #### 211MRR
    Spatial_211MRR = [0.0418, 0.0405, 0.0455, 0.0401, 0.0428, 0.0369, 0.0455, 0.0466, 0.039, 0.0501, 0.0447, 0.0441, 0.0455, 0.0609, 0.0359, 0.0396, 0.0468, 0.0487, 0.0429, 0.048, 0.0571, 0.0472, 0.0523, 0.0621, 0.0618, 0.0501, 0.0601, 0.0489, 0.0518, 0.0485, 0.0312, 0.0426, 0.0397, 0.0373, 0.0365, 0.0392, 0.0417, 0.0461, 0.0506, 0.0544, 0.065, 0.0489, 0.052, 0.041, 0.043, 0.0403, 0.0397, 0.0487, 0.0526, 0.0571, 0.0421, 0.0483, 0.0409, 0.0523, 0.034, 0.0483, 0.04, 0.0392, 0.0385, 0.0368, 0.0491, 0.0509, 0.0663, 0.0429, 0.0542, 0.0551, 0.0514, 0.0584, 0.0452, 0.0446, 0.0433, 0.0455, 0.0519, 0.0447, 0.0412, 0.0548, 0.0542, 0.047, 0.0524, 0.0479, 0.0585, 0.0569, 0.0541, 0.0407, 0.06, 0.0507, 0.0332, 0.0287, 0.0353, 0.0426, 0.0467, 0.0564, 0.0323, 0.0288, 0.0324, 0.0432, 0.0341, 0.0346, 0.0334, 0.0381, 0.0412, 0.0331, 0.0394, 0.0333, 0.036, 0.0434, 0.0384, 0.0365, 0.0292, 0.034, 0.0308, 0.0354, 0.0375, 0.0324, 0.0529, 0.0779, 0.0535, 0.0581, 0.0604, 0.0577, 0.0352, 0.056, 0.0518, 0.054, 0.0421, 0.0566, 0.0371, 0.0446, 0.0445, 0.0438, 0.0531, 0.0441, 0.071, 0.0655, 0.0383, 0.0374, 0.0408, 0.0523, 0.0526, 0.0341, 0.0504, 0.0676, 0.0512, 0.0311, 0.0426, 0.0374, 0.0453, 0.0467, 0.0486, 0.0453, 0.0408, 0.041, 0.0321, 0.0354, 0.0477, 0.0382, 0.029, 0.0344, 0.0296, 0.0321, 0.0295, 0.0395, 0.0472, 0.0337, 0.029, 0.0243, 0.0282, 0.0292, 0.0322, 0.0324, 0.0361, 0.0364, 0.036, 0.0295, 0.0399, 0.0374, 0.0336, 0.026, 0.0406, 0.0681, 0.0244, 0.0556, 0.0268, 0.0455, 0.0386, 0.0333, 0.0407, 0.0379, 0.0397, 0.0326, 0.0361, 0.076, 0.0363, 0.0306, 0.0426, 0.035, 0.0413, 0.0353, 0.0382, 0.0352, 0.0367, 0.0464, 0.048, 0.0468, 0.0487, 0.0362, 0.0305, 0.0598, 0.0348, 0.0432, 0.0468, 0.0505, 0.0558, 0.0462, 0.0484, 0.0375, 0.0408, 0.0508, 0.0538, 0.0591, 0.0432, 0.0382, 0.0412, 0.0347, 0.0468, 0.0752, 0.0444, 0.0484, 0.0408, 0.0387, 0.0359, 0.0543, 0.0426, 0.0428, 0.0659, 0.0499, 0.0401, 0.0632, 0.051, 0.0438, 0.0339, 0.0398, 0.0511, 0.0534, 0.0321, 0.0427, 0.044, 0.0433, 0.0453, 0.0557, 0.0456, 0.0643, 0.0648, 0.0625, 0.0435, 0.0638, 0.0577, 0.0526, 0.0484, 0.0451, 0.0504, 0.0563, 0.052, 0.0762, 0.0362, 0.0342, 0.0425, 0.0445, 0.0374, 0.0375, 0.0412, 0.0456, 0.053, 0.049, 0.0529, 0.056, 0.0526, 0.0406, 0.0472, 0.0415, 0.056, 0.0532, 0.0384, 0.0574, 0.0588, 0.0375, 0.0403, 0.0409, 0.0341, 0.0499, 0.0446, 0.0462, 0.0571, 0.0602, 0.0475, 0.0434]

    #### 218MN
    Spatial_218MN = [0.0735, 0.0326, 0.0268, 0.073, 0.0309, 0.0501, 0.0307, 0.0312, 0.053, 0.0379, 0.036, 0.0309, 0.0362, 0.0822, 0.0476, 0.0389, 0.0312, 0.0651, 0.0389, 0.0327, 0.0332, 0.0301, 0.0299, 0.0615, 0.0366, 0.0385, 0.0327, 0.0337, 0.0346, 0.0344, 0.0328, 0.034, 0.0356, 0.0332, 0.037, 0.059, 0.0307, 0.0328, 0.0305, 0.0347, 0.0341, 0.034, 0.0315, 0.05, 0.0368, 0.0372, 0.0335, 0.0411, 0.038, 0.0364, 0.035, 0.036, 0.0772, 0.0424, 0.0374, 0.0366, 0.0308, 0.0357, 0.035, 0.0346, 0.0388, 0.0393, 0.0353, 0.0292, 0.0346, 0.0368, 0.0371, 0.0379, 0.039, 0.0342, 0.0464, 0.0427, 0.0756, 0.0347, 0.0413, 0.0426, 0.0371, 0.0483, 0.0377, 0.0358, 0.0345, 0.0292, 0.043, 0.0725, 0.0424, 0.0359, 0.0325, 0.0462, 0.0491, 0.0458, 0.0454, 0.0474, 0.0463, 0.0579, 0.0508, 0.0445, 0.0465, 0.0572, 0.0486, 0.0716, 0.0389, 0.0363, 0.0366, 0.041, 0.0367, 0.0361, 0.039, 0.0367, 0.0515, 0.0453, 0.0482, 0.0492, 0.0539, 0.0576, 0.0537, 0.0445, 0.0497, 0.0501, 0.0554, 0.0564, 0.0363, 0.0311, 0.0342, 0.032, 0.0321, 0.0427, 0.0381, 0.0362, 0.04, 0.0288, 0.0296, 0.0347, 0.0301, 0.0301, 0.0299, 0.0301, 0.033, 0.0326, 0.0343, 0.0323, 0.0848, 0.0336, 0.0324, 0.0335, 0.0394, 0.0308, 0.0439, 0.0538, 0.0497, 0.0391, 0.0525, 0.0538, 0.0613, 0.0645, 0.0271, 0.0579, 0.0336, 0.0363, 0.0384, 0.0365, 0.0351, 0.0374, 0.0409, 0.0411, 0.0421, 0.0464, 0.0563, 0.0578, 0.0428, 0.0469, 0.0518, 0.0369, 0.0381, 0.0476, 0.0371, 0.0415, 0.0518, 0.0427, 0.0406, 0.036, 0.0498, 0.0592, 0.05, 0.0548, 0.0633, 0.0781, 0.0442, 0.0446, 0.049, 0.071, 0.0631, 0.0378, 0.0577, 0.0538, 0.0387, 0.0467, 0.0349, 0.0367, 0.0403, 0.0367, 0.0551, 0.0615, 0.0554, 0.0418, 0.04, 0.0461, 0.0423, 0.0417, 0.0316, 0.0447, 0.034, 0.038, 0.0523, 0.0474, 0.0559, 0.0505, 0.05, 0.0523, 0.0348, 0.0357, 0.0425, 0.0523, 0.0548, 0.0584, 0.0494, 0.049, 0.0357, 0.0353, 0.0751, 0.0453, 0.05, 0.052, 0.0615, 0.0588, 0.0507, 0.0588, 0.0586, 0.0564, 0.0499, 0.0597, 0.0506, 0.0517, 0.0525, 0.041, 0.0553, 0.0367, 0.0372, 0.0427, 0.061, 0.0539, 0.0356, 0.0403, 0.0398, 0.0487, 0.0405, 0.0335, 0.0536, 0.0414, 0.0399, 0.0543, 0.0425]

    #### 21ML
    Spatial_21ML = [0.0297, 0.0648, 0.0348, 0.0425, 0.0503, 0.0756, 0.0396, 0.0811, 0.0305, 0.0483, 0.0352, 0.0345, 0.0354, 0.0783, 0.044, 0.0259, 0.0345, 0.0492, 0.0274, 0.0813, 0.0779, 0.0383, 0.026, 0.041, 0.0355, 0.0269, 0.0597, 0.0258, 0.0398, 0.0446, 0.0428, 0.0241, 0.0781, 0.0434, 0.0592, 0.0806, 0.0583, 0.0322, 0.0781, 0.0745, 0.0572, 0.0419, 0.0317, 0.0216, 0.0363, 0.0301, 0.0437, 0.0259, 0.0304, 0.0461, 0.0355, 0.0293]

    Spatial = Spatial_206FRL + Spatial_204FR + Spatial_203MN + Spatial_187FN + Spatial_211MRR + Spatial_63MR + Spatial_218MN + Spatial_54MRL + Spatial_21ML
    return Spatial

def data_sigma_6():
    #### 54MRL
    Spatial_54MRL = [0.0353, 0.0378, 0.037, 0.0708, 0.0579, 0.0605, 0.0331, 0.0529, 0.0555, 0.08, 0.0746, 0.0512, 0.056, 0.0624, 0.043, 0.0657, 0.0657, 0.0346, 0.0624, 0.0508, 0.0422, 0.0348, 0.0514, 0.0431, 0.0316, 0.042, 0.0883, 0.0717, 0.0492, 0.0378, 0.0524, 0.0854, 0.0437, 0.0512, 0.034, 0.0474, 0.0832, 0.0468, 0.0442, 0.0602, 0.0396, 0.0566, 0.052, 0.0799, 0.0456, 0.0412, 0.0362, 0.0469, 0.0362, 0.0668, 0.0531, 0.1026, 0.041, 0.0552, 0.0612, 0.0345, 0.0377, 0.0436, 0.0592, 0.0551, 0.0378, 0.0407, 0.0592, 0.0594, 0.0666, 0.0403, 0.0524, 0.0519, 0.0482, 0.0517, 0.0565, 0.0422, 0.0746, 0.0739, 0.0572, 0.0721, 0.0613, 0.0637, 0.0719, 0.0466, 0.0501, 0.0427, 0.0658, 0.0491, 0.0622, 0.0506, 0.0824, 0.0508, 0.0413, 0.0518, 0.0605, 0.0487, 0.0315, 0.0648, 0.0512, 0.0839, 0.0735, 0.0582, 0.0415, 0.0458, 0.0755, 0.0405, 0.052, 0.0323, 0.0709, 0.0488, 0.0545, 0.0428, 0.0547, 0.0361, 0.0356, 0.0396, 0.0677, 0.0379, 0.0925, 0.041, 0.0361, 0.0373, 0.0567, 0.0438, 0.0368, 0.0357, 0.0331, 0.0771, 0.0428, 0.0414, 0.0523, 0.034, 0.0634, 0.0382, 0.0629, 0.0416, 0.0376, 0.039, 0.0691, 0.074, 0.0411, 0.0336, 0.0471, 0.0787, 0.056, 0.042, 0.0395, 0.0668, 0.0692, 0.0329, 0.0546, 0.0496, 0.0564, 0.0721, 0.0364, 0.0386, 0.0482, 0.0374, 0.0622, 0.0323, 0.039, 0.0571, 0.0407, 0.032, 0.0476, 0.067, 0.0681, 0.0287, 0.0404, 0.0397, 0.0409, 0.0711, 0.0818, 0.0447, 0.0542, 0.0899, 0.0441, 0.0449, 0.0351, 0.0631, 0.0514, 0.0814, 0.0492, 0.0523, 0.0341, 0.0338, 0.0779, 0.0386, 0.0382, 0.0349, 0.0369, 0.0338, 0.0912, 0.0683, 0.0602, 0.0433, 0.0386, 0.0399, 0.053, 0.059, 0.0478, 0.0581, 0.0373, 0.0799, 0.0566, 0.0643, 0.0845, 0.0554, 0.0333, 0.0387, 0.0535, 0.0325, 0.0847, 0.0456, 0.032, 0.0397, 0.0442, 0.0363, 0.0951, 0.0397, 0.0337, 0.0314, 0.0312, 0.0317, 0.0609, 0.054, 0.0395, 0.0729, 0.0444, 0.0538, 0.0537, 0.0807, 0.0627, 0.0736, 0.0876, 0.0377, 0.0846, 0.0421, 0.0472, 0.0392, 0.0867, 0.0368, 0.0381, 0.0361, 0.0406, 0.0529, 0.0427, 0.0482, 0.0465, 0.049, 0.0486, 0.0554, 0.0481, 0.0483, 0.071, 0.0715, 0.0417, 0.0377, 0.0637, 0.0504, 0.0682, 0.0463, 0.056, 0.0569, 0.0587, 0.0457, 0.0612, 0.0409, 0.0418, 0.0494, 0.0682, 0.0596, 0.0392, 0.0448, 0.0451, 0.0543, 0.0446, 0.037, 0.0589, 0.0463, 0.0445, 0.0592, 0.0464]

    #### 63MR
    Spatial_63MR = [0.0491, 0.0296, 0.0288, 0.0327, 0.0478, 0.063, 0.0456, 0.0269, 0.0455, 0.046, 0.0296, 0.038, 0.0457, 0.0314, 0.0508, 0.0416, 0.0723, 0.0321, 0.0319, 0.0734, 0.0375, 0.0356, 0.0337, 0.0497, 0.0303, 0.0461, 0.0313, 0.0301, 0.063, 0.0641, 0.0294, 0.0345, 0.0389, 0.0412, 0.0537, 0.0381, 0.0739, 0.035, 0.0503, 0.0661, 0.0481, 0.0408, 0.0371, 0.0269, 0.0348, 0.0323, 0.0335, 0.0404, 0.0519, 0.0389, 0.0308, 0.0313, 0.0423, 0.0621, 0.0614, 0.0434, 0.0486, 0.0314, 0.0735, 0.0361, 0.0522, 0.043, 0.0386, 0.0663, 0.0886, 0.0339, 0.0514, 0.035, 0.0378, 0.0473, 0.0499, 0.0452, 0.0371, 0.0635, 0.0633, 0.0359, 0.0743, 0.0426, 0.0406, 0.0471, 0.0636, 0.0482, 0.0453, 0.0348, 0.0285, 0.0274, 0.0284, 0.0579, 0.0316, 0.0298, 0.0335, 0.0841, 0.069, 0.0403, 0.0581, 0.0283, 0.0641, 0.0395, 0.0513, 0.0867, 0.0449, 0.0596, 0.049, 0.072, 0.0486, 0.0555, 0.0602, 0.0678, 0.0463, 0.0482, 0.0565, 0.0287, 0.0332, 0.0408, 0.0394, 0.0293, 0.0409, 0.0296, 0.044, 0.0553, 0.0605, 0.052, 0.0336, 0.0618, 0.0846, 0.1026, 0.0258, 0.037, 0.034, 0.0294, 0.0347, 0.0412, 0.055, 0.0736, 0.0664, 0.0383, 0.0291, 0.0639, 0.0262, 0.0791, 0.0518, 0.0421, 0.0751, 0.0891, 0.0434, 0.0472, 0.0307, 0.0847, 0.0413, 0.0646, 0.0402, 0.0536, 0.0674, 0.0348, 0.0432, 0.0341, 0.0569, 0.0685, 0.0614, 0.0656, 0.0568, 0.0327, 0.0355, 0.0301, 0.0382, 0.0464, 0.0346, 0.0862, 0.0417, 0.0308, 0.0679, 0.0333, 0.0403, 0.038, 0.0428, 0.0554, 0.0344, 0.0416, 0.104, 0.0334, 0.035, 0.0425, 0.0394, 0.0367, 0.0502, 0.032, 0.0804, 0.0485, 0.0401, 0.0411, 0.0331, 0.0324, 0.0352, 0.0364, 0.0387, 0.0518, 0.0724, 0.0375, 0.0793, 0.0452, 0.0536, 0.0212, 0.0694, 0.059, 0.0597, 0.0895, 0.0389, 0.0261, 0.0625, 0.0661, 0.049]

    #### 187FN
    Spatial_187FN = [0.0442, 0.0369, 0.0442, 0.0878, 0.0586, 0.0836, 0.0399, 0.0349, 0.033, 0.0393, 0.0341, 0.037, 0.0349, 0.0398, 0.0407, 0.0417, 0.0353, 0.044, 0.0396, 0.0412, 0.0381, 0.0421, 0.0371, 0.0512, 0.0387, 0.0411, 0.0368, 0.0401, 0.0638, 0.0358, 0.034, 0.0391, 0.0458, 0.0311, 0.036, 0.0396, 0.0376, 0.0398, 0.0345, 0.0372, 0.043, 0.0336, 0.0355, 0.0344, 0.0546, 0.0643, 0.0527, 0.0766, 0.0604, 0.077, 0.053, 0.0412, 0.0685, 0.0526, 0.047, 0.0674, 0.0462, 0.0549, 0.0587, 0.0533, 0.0423, 0.0485, 0.0394, 0.0549, 0.0454, 0.0463, 0.0472, 0.0575, 0.0441, 0.0423, 0.063, 0.0589, 0.0453, 0.0579, 0.055, 0.0656, 0.0631, 0.0575, 0.0542, 0.0524, 0.0638, 0.0512, 0.0417, 0.0518, 0.054, 0.0401, 0.0673, 0.057, 0.069, 0.0978, 0.0516, 0.0547, 0.0521, 0.0464, 0.0802, 0.0626, 0.065, 0.0679, 0.0812, 0.0635, 0.0795, 0.0415, 0.0501, 0.0407, 0.0433, 0.0551, 0.0581, 0.0357, 0.0327, 0.0381, 0.0419, 0.0388, 0.0493, 0.0407, 0.0406, 0.0458, 0.0456, 0.0637, 0.0304, 0.0398, 0.0498, 0.0536, 0.0378, 0.055, 0.046, 0.0594, 0.0599, 0.0422, 0.0596, 0.1107, 0.04, 0.0383, 0.0393, 0.0323, 0.0286, 0.0524, 0.0487, 0.0574, 0.0453, 0.0692, 0.0554, 0.06, 0.0402, 0.0472, 0.0619, 0.0513, 0.0581, 0.0648, 0.0639, 0.0506, 0.04, 0.0424, 0.0413, 0.0563, 0.0485, 0.0479, 0.0639, 0.0586, 0.0548, 0.0406, 0.0482, 0.0361, 0.0533, 0.0452, 0.0567, 0.0528, 0.0613, 0.0663, 0.051, 0.0678, 0.0658, 0.04, 0.0756, 0.0352, 0.0539, 0.0457, 0.0456, 0.0433, 0.0502, 0.0746, 0.0771, 0.0644, 0.0576, 0.069, 0.0714, 0.0535, 0.0763, 0.0473, 0.0686, 0.0715, 0.0376, 0.042, 0.0518, 0.0387, 0.0838, 0.0535, 0.0796, 0.0478, 0.0414, 0.0451, 0.0554, 0.0427, 0.0401, 0.0556, 0.1037, 0.0495, 0.0478, 0.0545, 0.0744, 0.0471, 0.0494, 0.0632, 0.0348, 0.0491, 0.0495, 0.0565, 0.0382, 0.0463, 0.0429, 0.0421, 0.05, 0.051, 0.0422, 0.0414, 0.0698, 0.0515, 0.0595, 0.0533, 0.041, 0.053, 0.0545, 0.0614, 0.0668, 0.0621, 0.0673, 0.038, 0.0475, 0.0443, 0.0626, 0.0417, 0.0414, 0.0386, 0.047, 0.0393, 0.0392, 0.0487, 0.0642, 0.0574, 0.0684, 0.0606, 0.0524, 0.0384, 0.0389, 0.1041, 0.0397, 0.0384, 0.0446, 0.0523, 0.0479, 0.0398, 0.0411, 0.0312, 0.0483, 0.0625, 0.0401, 0.0637, 0.0547, 0.0476, 0.0606, 0.0568, 0.0563, 0.0935, 0.078, 0.0525, 0.0461, 0.0525, 0.0427, 0.0448, 0.0638, 0.0424, 0.0389, 0.0519, 0.0441, 0.0529, 0.0595, 0.0611, 0.0559, 0.0497, 0.0435, 0.0456, 0.0526, 0.0534, 0.0619, 0.0916, 0.0378, 0.0444, 0.0387, 0.0697, 0.0481, 0.0449, 0.0561, 0.0549, 0.0631, 0.0561, 0.0768, 0.0642, 0.0603, 0.0499, 0.0729, 0.0627, 0.0571, 0.0876, 0.0707, 0.0653, 0.0403, 0.0477, 0.075, 0.097, 0.051, 0.0397, 0.0332, 0.0631, 0.0385, 0.0373, 0.0423, 0.0449, 0.0408, 0.0374, 0.0539, 0.0456, 0.0601, 0.0326, 0.0416, 0.0399, 0.0467, 0.0838, 0.042, 0.048, 0.0873, 0.0596, 0.0379, 0.0525, 0.0507, 0.0504, 0.0536, 0.0607, 0.0673, 0.0773, 0.0488, 0.0841, 0.0484, 0.0949, 0.0343, 0.0446, 0.0386, 0.0423, 0.0407, 0.0359, 0.0401, 0.0402, 0.0455, 0.0984, 0.0828, 0.06, 0.0686, 0.0342, 0.0288, 0.0479, 0.0389, 0.0357, 0.0403, 0.0328, 0.0718, 0.035, 0.0339, 0.1022, 0.0458, 0.0358, 0.0465, 0.0323, 0.0363, 0.0392, 0.052, 0.0379, 0.0387]

    #### 203MN
    Spatial_203MN = [0.035, 0.04, 0.0454, 0.0329, 0.0332, 0.0352, 0.0356, 0.0351, 0.0456, 0.0366, 0.0332, 0.045, 0.048, 0.0348, 0.0831, 0.0444, 0.0829, 0.0393, 0.0369, 0.0362, 0.0437, 0.0431, 0.0369, 0.0339, 0.0516, 0.0404, 0.0414, 0.0373, 0.0399, 0.0374, 0.0385, 0.0404, 0.0382, 0.0392, 0.0606, 0.0445, 0.0446, 0.0495, 0.0557, 0.0426, 0.0486, 0.0542, 0.0456, 0.0474, 0.0474, 0.0551, 0.0477, 0.0502, 0.0544, 0.0556, 0.05, 0.0449, 0.0543, 0.0515, 0.0522, 0.0444, 0.041, 0.0416, 0.0527, 0.0514, 0.0484, 0.0488, 0.0464, 0.0935, 0.0544, 0.0516, 0.0416, 0.0461, 0.0464, 0.0545, 0.0538, 0.0516, 0.0497, 0.0482, 0.0532, 0.0554, 0.0493, 0.0446, 0.0621, 0.0462, 0.0499, 0.0474, 0.042, 0.0751, 0.0416, 0.0489, 0.0467, 0.0465, 0.0473, 0.0588, 0.0353, 0.0337, 0.0374, 0.0374, 0.0376, 0.0515, 0.0408, 0.0435, 0.0363, 0.0366, 0.0379, 0.0368, 0.0485, 0.0964, 0.0342, 0.0322, 0.0489, 0.0327, 0.0503, 0.0436, 0.0388, 0.037, 0.0344, 0.0389, 0.0334, 0.0329, 0.0419, 0.0477, 0.0362, 0.0374, 0.0383, 0.0505, 0.04, 0.0393, 0.0351, 0.0335, 0.0951, 0.0645, 0.0471, 0.0326, 0.0332, 0.0389, 0.0395, 0.0444, 0.0494, 0.0498, 0.049, 0.0468, 0.0468, 0.0485, 0.0443, 0.0468, 0.0445, 0.0378, 0.0493, 0.0587, 0.0485, 0.0467, 0.0501, 0.0473, 0.0427, 0.0496, 0.045, 0.0465, 0.0391, 0.0555, 0.0469, 0.0496, 0.0585, 0.0547, 0.0551, 0.0426, 0.0486, 0.039, 0.0949, 0.1012, 0.0479, 0.0379, 0.0436, 0.0465, 0.0428, 0.0474, 0.0471, 0.04, 0.0528, 0.0437, 0.06, 0.0447, 0.0492, 0.0394, 0.0494, 0.0525, 0.0386, 0.0442, 0.0543, 0.0745, 0.0548, 0.0546, 0.0941, 0.0387, 0.054, 0.0445, 0.0376, 0.0408, 0.0387, 0.0609, 0.046, 0.0374, 0.0421, 0.0428, 0.0391, 0.0427, 0.0425, 0.0411, 0.046, 0.0421, 0.04, 0.0554, 0.0359, 0.037, 0.0395, 0.0396, 0.042, 0.04, 0.047, 0.0435, 0.0556, 0.0551, 0.0475, 0.041, 0.0473, 0.0493, 0.0755, 0.0369, 0.055, 0.0371, 0.0401, 0.0426, 0.0411, 0.0426, 0.0375, 0.0377, 0.0455, 0.0414, 0.0472, 0.0371, 0.0496, 0.047, 0.038, 0.0369, 0.0335, 0.0359, 0.0423, 0.0387, 0.043, 0.0699, 0.0371, 0.0352, 0.0394, 0.0386, 0.0389, 0.0333, 0.0363, 0.0359, 0.0404, 0.0587, 0.0392, 0.0361, 0.106, 0.0619, 0.0739, 0.0484, 0.0421, 0.0303, 0.034, 0.0416, 0.0444, 0.04, 0.048, 0.0388, 0.0504, 0.0393, 0.0402, 0.0827, 0.0599, 0.0426, 0.0398, 0.0461, 0.0384, 0.0426, 0.0526, 0.0511, 0.051, 0.0455, 0.0513, 0.0531, 0.0386, 0.0392, 0.0455, 0.033, 0.0552, 0.0442, 0.0561, 0.0826, 0.0415, 0.0411, 0.0387, 0.0372, 0.0601, 0.0403, 0.0593, 0.0637, 0.0987, 0.0541, 0.0394, 0.0326, 0.0444, 0.039, 0.0555, 0.0456, 0.0468, 0.05, 0.0429, 0.0503, 0.0575, 0.0463, 0.0579, 0.0398, 0.0363, 0.0322, 0.0315, 0.0954, 0.0355, 0.0405, 0.0437, 0.044, 0.0435, 0.0405, 0.0442, 0.0451, 0.0472, 0.0659]

    #### 204FR
    Spatial_204FR = [0.0309, 0.0855, 0.035, 0.0721, 0.0751, 0.0596, 0.0297, 0.031, 0.0327, 0.0364, 0.0391, 0.0313, 0.0695, 0.0411, 0.0417, 0.0435, 0.0387, 0.0855, 0.0545, 0.0603, 0.0795, 0.0493, 0.0449, 0.0349, 0.0337, 0.035, 0.041, 0.0455, 0.0307, 0.0369, 0.0851, 0.0549, 0.0799, 0.0635, 0.0384, 0.0362, 0.038, 0.0367, 0.0389, 0.0393, 0.034, 0.0576, 0.0389, 0.0338, 0.0366, 0.0505, 0.0326, 0.0443, 0.0416, 0.0444, 0.0442, 0.0408, 0.046, 0.045, 0.0509, 0.0349, 0.042, 0.0387, 0.0393, 0.0939, 0.0437, 0.0429, 0.0441, 0.0479, 0.047, 0.045, 0.043, 0.0384, 0.0517, 0.0468, 0.0449, 0.0476, 0.048, 0.0597, 0.0451, 0.0464, 0.0982, 0.0827, 0.0693, 0.0411, 0.0409, 0.0469, 0.0481, 0.0462, 0.0663, 0.0439, 0.0432, 0.0466, 0.0503, 0.0505, 0.0513, 0.0519, 0.0432, 0.0436, 0.0404, 0.0789, 0.0509, 0.0355, 0.0387, 0.0468, 0.0611, 0.0567, 0.0655, 0.0594, 0.059, 0.0656, 0.0601, 0.059, 0.0674, 0.0586, 0.0561, 0.0666, 0.0645, 0.0617, 0.0611, 0.0609, 0.0624, 0.0601, 0.0499, 0.0523, 0.0627, 0.0562, 0.0631, 0.0638, 0.0569, 0.062, 0.0612, 0.0614, 0.0547, 0.0686, 0.049, 0.0463, 0.0508, 0.0497, 0.0493, 0.0459, 0.0441, 0.044, 0.0382, 0.042, 0.041, 0.0373, 0.0365, 0.0556, 0.0605, 0.0701, 0.0455, 0.0426, 0.048, 0.0452, 0.0504, 0.0404, 0.0457, 0.047, 0.1017, 0.0837, 0.0383, 0.0835, 0.0453, 0.0431, 0.0416, 0.0436, 0.0526, 0.0459, 0.0507, 0.0439, 0.0475, 0.0486, 0.047, 0.0509, 0.0479, 0.0553, 0.0522, 0.0419, 0.042, 0.0404, 0.0508, 0.0421, 0.0709, 0.0807, 0.0486, 0.0454, 0.044, 0.044, 0.0525, 0.0525, 0.051, 0.0534, 0.0508, 0.0484, 0.049, 0.0525, 0.0468, 0.0494, 0.0433, 0.0538, 0.0434, 0.041, 0.0447, 0.0385, 0.0461, 0.0393, 0.0529, 0.0434, 0.0425, 0.0355, 0.0462, 0.051, 0.0763, 0.0517, 0.0536, 0.0522, 0.0473, 0.0374, 0.058, 0.0401, 0.0551, 0.0669, 0.0357, 0.0444, 0.0448, 0.0479, 0.0497, 0.054, 0.0544, 0.0511, 0.0467, 0.0418, 0.0504, 0.0529, 0.0541, 0.0515, 0.0497, 0.0468, 0.0416, 0.0507, 0.0591, 0.0642, 0.045, 0.0557, 0.0546, 0.0422, 0.0397, 0.0448, 0.0512, 0.0497, 0.0421, 0.0436, 0.0613, 0.0611, 0.0931, 0.0763, 0.041, 0.0366, 0.0414, 0.0423, 0.0447, 0.0563, 0.05, 0.0525, 0.0491, 0.0621, 0.0684, 0.0384, 0.0592, 0.0477, 0.0461, 0.0454, 0.0567, 0.0559, 0.069, 0.0402, 0.0626, 0.06, 0.0554, 0.0421, 0.0375, 0.0376, 0.0366, 0.0465, 0.0703, 0.0713, 0.0625, 0.057, 0.0455, 0.0468, 0.0482, 0.0499]

    #### 206FRL
    Spatial_206FRL = [0.0422, 0.042, 0.052, 0.0443, 0.051, 0.037, 0.0448, 0.0586, 0.0369, 0.0415, 0.0386, 0.0402, 0.039, 0.0398, 0.0539, 0.0395, 0.0403, 0.0403, 0.034, 0.0414, 0.0409, 0.0414, 0.048, 0.0397, 0.0335, 0.0402, 0.0394, 0.0376, 0.0378, 0.071, 0.0563, 0.0356, 0.0377, 0.048, 0.0478, 0.0371, 0.035, 0.0348, 0.0383, 0.0502, 0.0392, 0.0324, 0.0476, 0.0433, 0.0347, 0.0392, 0.0345, 0.0371, 0.0304, 0.0338, 0.0374, 0.0341, 0.0344, 0.0608, 0.0654, 0.0369, 0.0735, 0.0616, 0.0328, 0.0357, 0.0437, 0.0433, 0.0786, 0.0508, 0.0393, 0.0371, 0.0381, 0.0408, 0.0368, 0.0379, 0.0393, 0.041, 0.0384, 0.0375, 0.0399, 0.0568, 0.0396, 0.044, 0.0366, 0.0403, 0.0381, 0.0374, 0.0382, 0.0433, 0.0505, 0.0357, 0.0381, 0.085, 0.0718, 0.0754, 0.0651, 0.0708, 0.037, 0.0432, 0.04, 0.0424, 0.0413, 0.0358, 0.035, 0.0434, 0.0421, 0.0379, 0.0477, 0.0401, 0.0458, 0.065, 0.043, 0.0421, 0.04, 0.0432, 0.0415, 0.0398, 0.0361, 0.0426, 0.08, 0.0608, 0.0459, 0.0489, 0.0465, 0.0388, 0.0515, 0.0485, 0.0494, 0.0478, 0.0536, 0.051, 0.0476, 0.0601, 0.0542, 0.0548, 0.0462, 0.054, 0.0455, 0.0478, 0.0558, 0.0504, 0.0502, 0.0722, 0.0423, 0.0464, 0.0445, 0.0463, 0.046, 0.0472, 0.0494, 0.0485, 0.0483, 0.048, 0.0473, 0.0473, 0.0465, 0.0439, 0.046, 0.0473, 0.0459, 0.0462, 0.0371, 0.0443, 0.042, 0.0407, 0.0404, 0.0467, 0.0414, 0.0398, 0.0405, 0.0466, 0.0419, 0.0429, 0.0455, 0.0572, 0.0346, 0.0366, 0.0386, 0.0426, 0.0432, 0.0411, 0.0422, 0.0438, 0.0437, 0.0431, 0.0404, 0.0401, 0.0416, 0.0412, 0.0402, 0.0412, 0.0468, 0.0391, 0.0402, 0.0417, 0.0669, 0.0513, 0.0529, 0.0378, 0.0407, 0.0409, 0.0443, 0.0941, 0.0584, 0.0593, 0.0489, 0.0411, 0.0381, 0.0419, 0.0428, 0.0446, 0.0493, 0.0393, 0.0341, 0.0453, 0.0609, 0.0386, 0.0412, 0.0486, 0.0468, 0.0455, 0.0468, 0.0493, 0.0604, 0.0412, 0.0456, 0.0451, 0.0479, 0.0452, 0.0469, 0.0519, 0.0491, 0.0483, 0.0461, 0.0447, 0.0554, 0.0537, 0.0542, 0.0601, 0.0657, 0.038, 0.0422, 0.0466, 0.0487, 0.0422, 0.046, 0.0503, 0.0418, 0.045, 0.0454, 0.0387, 0.0425, 0.0416, 0.04, 0.0412, 0.0417, 0.0386, 0.0415, 0.041, 0.0954, 0.0782, 0.0602, 0.0471, 0.0418, 0.0447, 0.0426, 0.0353, 0.0558, 0.051, 0.0465, 0.0473, 0.0435, 0.0743, 0.0399, 0.0423, 0.0517, 0.0529, 0.0529, 0.0539, 0.0516, 0.0489, 0.0518, 0.0493, 0.0432, 0.0447, 0.0438, 0.053, 0.0427, 0.0544, 0.0445, 0.0366, 0.0402, 0.042, 0.0504, 0.0457, 0.0493, 0.0423, 0.0383, 0.0732, 0.0715, 0.0399, 0.0397, 0.0433, 0.0404, 0.0426, 0.0397, 0.036, 0.0416, 0.0398, 0.0558, 0.0624, 0.0509, 0.0502, 0.0496, 0.0469, 0.0449, 0.0522, 0.0533, 0.048, 0.0463, 0.045, 0.045, 0.0506, 0.0515, 0.0591, 0.0645, 0.0668, 0.0674, 0.0786, 0.0737, 0.0656, 0.0632, 0.0727, 0.0708, 0.0788, 0.064]

    #### 211MRR
    Spatial_211MRR = [0.0462, 0.0447, 0.051, 0.0445, 0.0482, 0.0407, 0.0509, 0.0522, 0.0433, 0.0562, 0.0497, 0.0493, 0.051, 0.0674, 0.0395, 0.0436, 0.0525, 0.0548, 0.048, 0.0535, 0.0647, 0.053, 0.0589, 0.0702, 0.0696, 0.0569, 0.068, 0.0542, 0.0576, 0.0548, 0.0332, 0.0467, 0.0436, 0.0409, 0.04, 0.0433, 0.0464, 0.0516, 0.057, 0.0615, 0.0743, 0.055, 0.0586, 0.0452, 0.0486, 0.0455, 0.0449, 0.0551, 0.0593, 0.0645, 0.0476, 0.0542, 0.0459, 0.0589, 0.0376, 0.0538, 0.0448, 0.0442, 0.0427, 0.0412, 0.0556, 0.0573, 0.0746, 0.0484, 0.061, 0.0622, 0.0584, 0.0664, 0.0507, 0.0504, 0.0485, 0.0512, 0.0588, 0.0497, 0.0471, 0.0621, 0.061, 0.0534, 0.0589, 0.054, 0.066, 0.0642, 0.0611, 0.0468, 0.0679, 0.0577, 0.0372, 0.0319, 0.0395, 0.0482, 0.0533, 0.0635, 0.0361, 0.0318, 0.0358, 0.0492, 0.0382, 0.0389, 0.0377, 0.0427, 0.0469, 0.0369, 0.0442, 0.0377, 0.0405, 0.0502, 0.0429, 0.0406, 0.0325, 0.0381, 0.0346, 0.0403, 0.0425, 0.0366, 0.0599, 0.0863, 0.0602, 0.0654, 0.0678, 0.0646, 0.039, 0.0628, 0.0584, 0.0617, 0.0482, 0.0622, 0.0404, 0.0498, 0.0498, 0.0494, 0.0599, 0.0494, 0.0806, 0.0737, 0.0425, 0.0414, 0.0462, 0.0588, 0.0586, 0.0376, 0.0556, 0.0741, 0.0568, 0.0344, 0.0473, 0.0418, 0.0506, 0.0517, 0.0528, 0.0514, 0.0455, 0.0459, 0.036, 0.0397, 0.054, 0.0425, 0.0327, 0.0391, 0.0336, 0.0361, 0.0331, 0.0448, 0.0533, 0.0388, 0.0324, 0.0269, 0.0318, 0.0328, 0.0363, 0.0371, 0.041, 0.0411, 0.0405, 0.0337, 0.0456, 0.0428, 0.0377, 0.0299, 0.046, 0.0764, 0.0266, 0.0604, 0.0294, 0.0509, 0.0422, 0.0364, 0.0451, 0.0422, 0.044, 0.0358, 0.0404, 0.0832, 0.0401, 0.0335, 0.0477, 0.0388, 0.047, 0.0394, 0.0424, 0.0391, 0.0412, 0.0525, 0.0542, 0.0526, 0.055, 0.0408, 0.0335, 0.0661, 0.0379, 0.0486, 0.053, 0.0571, 0.0632, 0.0519, 0.0546, 0.0419, 0.0459, 0.0564, 0.0609, 0.0661, 0.0487, 0.0424, 0.0461, 0.0396, 0.0528, 0.0857, 0.0501, 0.0538, 0.0458, 0.0434, 0.0403, 0.0615, 0.0485, 0.0488, 0.0739, 0.0561, 0.0462, 0.0711, 0.0571, 0.0488, 0.0379, 0.0451, 0.0575, 0.0588, 0.0352, 0.0479, 0.0503, 0.0488, 0.0509, 0.0626, 0.0515, 0.0727, 0.073, 0.0703, 0.0487, 0.0714, 0.0645, 0.0591, 0.0549, 0.0513, 0.0578, 0.0642, 0.0576, 0.0833, 0.0397, 0.0383, 0.0478, 0.05, 0.0423, 0.0418, 0.0465, 0.0522, 0.0598, 0.0552, 0.0603, 0.0636, 0.06, 0.0458, 0.0537, 0.0472, 0.0628, 0.0603, 0.0434, 0.0648, 0.0654, 0.0422, 0.0454, 0.0463, 0.0379, 0.0566, 0.0496, 0.0519, 0.0648, 0.068, 0.054, 0.0496]

    #### 218MN
    Spatial_218MN = [0.081, 0.0363, 0.0297, 0.0799, 0.035, 0.056, 0.0346, 0.0352, 0.0594, 0.0418, 0.0407, 0.0342, 0.0407, 0.0884, 0.053, 0.0436, 0.0341, 0.0719, 0.0435, 0.0364, 0.037, 0.0337, 0.0333, 0.0669, 0.0415, 0.0431, 0.0367, 0.0379, 0.0388, 0.0377, 0.0367, 0.038, 0.04, 0.0375, 0.042, 0.0637, 0.0341, 0.0365, 0.0346, 0.0389, 0.0384, 0.0383, 0.0347, 0.0546, 0.0415, 0.042, 0.0375, 0.0461, 0.0425, 0.0409, 0.0392, 0.0402, 0.0836, 0.0462, 0.042, 0.0409, 0.0341, 0.0401, 0.0391, 0.0387, 0.0433, 0.0441, 0.0403, 0.0327, 0.0385, 0.0412, 0.0414, 0.0427, 0.0441, 0.038, 0.0525, 0.0483, 0.0811, 0.0385, 0.0464, 0.0479, 0.0415, 0.0543, 0.0423, 0.04, 0.0378, 0.0319, 0.0476, 0.0781, 0.0477, 0.0401, 0.0363, 0.0523, 0.0553, 0.0515, 0.0509, 0.0534, 0.0522, 0.0657, 0.0568, 0.0503, 0.0525, 0.0642, 0.0548, 0.0776, 0.0438, 0.0411, 0.0418, 0.0466, 0.041, 0.0403, 0.0442, 0.0409, 0.0581, 0.0515, 0.0541, 0.0554, 0.0605, 0.0663, 0.0608, 0.0501, 0.0563, 0.0569, 0.0626, 0.0629, 0.0413, 0.0348, 0.0381, 0.0363, 0.0363, 0.0485, 0.0433, 0.0408, 0.0446, 0.0318, 0.0328, 0.0378, 0.0338, 0.0339, 0.0336, 0.0336, 0.0374, 0.0368, 0.0385, 0.0363, 0.0917, 0.0376, 0.0372, 0.0378, 0.0445, 0.0346, 0.0502, 0.0606, 0.0558, 0.0441, 0.0578, 0.0603, 0.0665, 0.0703, 0.0298, 0.0629, 0.037, 0.0404, 0.0435, 0.0409, 0.0387, 0.0415, 0.0464, 0.0463, 0.0472, 0.0525, 0.0638, 0.0646, 0.0487, 0.053, 0.0574, 0.0411, 0.0426, 0.0523, 0.0414, 0.0467, 0.0588, 0.0478, 0.0454, 0.0402, 0.0562, 0.0666, 0.0563, 0.0623, 0.069, 0.0843, 0.0495, 0.0502, 0.0557, 0.0797, 0.0712, 0.0426, 0.065, 0.0606, 0.0436, 0.0522, 0.0393, 0.0416, 0.045, 0.0415, 0.0623, 0.0693, 0.0628, 0.0474, 0.0444, 0.0521, 0.0476, 0.0472, 0.0351, 0.0503, 0.0378, 0.0426, 0.0594, 0.0531, 0.063, 0.0575, 0.0566, 0.0578, 0.0388, 0.04, 0.0486, 0.0596, 0.062, 0.065, 0.0553, 0.0545, 0.0403, 0.0399, 0.0813, 0.0512, 0.0566, 0.059, 0.0702, 0.066, 0.0572, 0.0657, 0.0658, 0.0634, 0.0568, 0.0676, 0.0574, 0.0583, 0.0594, 0.0463, 0.0624, 0.0411, 0.0421, 0.0496, 0.0694, 0.0604, 0.0398, 0.0455, 0.0454, 0.0555, 0.0458, 0.0374, 0.0602, 0.0469, 0.0452, 0.0597, 0.0462]

    #### 21ML
    Spatial_21ML = [0.0329, 0.0721, 0.0384, 0.0472, 0.0553, 0.0833, 0.043, 0.087, 0.0338, 0.0521, 0.0389, 0.0376, 0.0392, 0.0853, 0.048, 0.0283, 0.0377, 0.0541, 0.0297, 0.0887, 0.084, 0.0413, 0.0281, 0.0449, 0.0394, 0.0292, 0.0653, 0.0281, 0.0437, 0.049, 0.0472, 0.0262, 0.086, 0.0474, 0.064, 0.0871, 0.0629, 0.0348, 0.0826, 0.0787, 0.0613, 0.0444, 0.0343, 0.0235, 0.0398, 0.033, 0.0469, 0.0288, 0.0329, 0.0508, 0.0385, 0.0321]

    Spatial = Spatial_206FRL + Spatial_204FR + Spatial_203MN + Spatial_187FN + Spatial_211MRR + Spatial_63MR + Spatial_218MN + Spatial_54MRL + Spatial_21ML
    return Spatial


sigma_funcs = {
    1: data_sigma_1,
    2: data_sigma_2,
    3: data_sigma_3,
    4: data_sigma_4,
    5: data_sigma_5
}

sigmas = [5,4,3,2,1]#sorted(sigma_funcs.keys())

import seaborn as sns
colors = sns.color_palette("colorblind", n_colors=5)
colors = [colors[4], colors[1], colors[2], colors[3], colors[0]]


for sigma, color in zip(sigmas, colors):
    Spatial = sigma_funcs[sigma]()  # Call the corresponding function
    mean_val = np.mean(Spatial)
    hist, bin_edges = np.histogram(Spatial, bins=42, range=(0, 0.084), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_percent = (hist / hist.sum()) * 100

    # Plot line and mean
    ax2.fill_between(bin_centers, hist_percent, alpha=0.7, color=color, label=fr'$\sigma$ = {sigma}')
    ax2.axvline(mean_val, color=color, linestyle='--', linewidth=1)

#ax2.set_title('Standard Deviation of All Events\nfor Gaussian Smoothing $\sigma$ = 1 to 5')
ax2.set_xlabel('Std', fontsize=10)
ax2.set_ylabel('% of Events', fontsize=10)
ax2.set_xlim([0,0.082])
ax2.set_ylim([0,25])
ax2.legend(
    frameon=False,       # removes the legend box
    loc='upper right',   # position (e.g., 'best', 'lower left', etc.)
    fontsize=4,         # font size
    ncol=1,              # number of columns
    handlelength=2     # length of the legend line
)

x_min, x_max = ax2.get_xlim()
y_min, y_max = ax2.get_ylim()
ax2.set_aspect((x_max - x_min) / (y_max - y_min), adjustable='box')

ax2.tick_params(axis='x', labelsize=8)
ax2.tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig('Basic patterns - Score VS Noise.pdf',bbox_inches = 'tight')#, dpi=300)

plt.show()



