import numpy as np
import matplotlib.pyplot as plt
from Algos.Create_Patterns import create_patterns, create_gaussians, create_gaussians_moving
from Algos.Data_Processing import Filter, resize, decrease_frame_rate, normalize_data
from Algos.Horn_Schunck import horn_schunck , horn_schunck_phase
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from Algos.Display import plot_quiver



def horn_schunck_flow(dff, frame, alpha=0.3, iterations=150, phase=False):

    image1 = dff[:, :, frame]
    image2 = dff[:, :, frame + 1]

    if phase:
        flow, convergence = horn_schunck_phase(image1, image2, alpha, iterations)
    else:
        flow, convergence = horn_schunck(image1, image2, alpha, iterations)

    return flow, convergence

def add_noise(dff, std, seed=32):
    rng = np.random.default_rng(seed)

    for t in range(dff.shape[2]):
        dff[:, :, t] += rng.normal(0, std, size=(dff.shape[0], dff.shape[1]))

def plot_wavefront_detection_4rows(datasets, alpha=0.3, iterations=150, rect_size=(34, 4), sigma=1, prominence=0, spacing=4, scale=0.04):

    if len(datasets) != 4:
        raise ValueError("datasets must contain exactly 4 entries.")

    fig, axes = plt.subplots(4, 4, figsize=(10, 9))

    for row, data in enumerate(datasets):

        dff = data["dff"]
        frame_idx = data["frame"]
        x = data["x"]
        y = data["y"]
        title = data.get("title", f"Dataset {row + 1}")


        flow, convergence = horn_schunck_flow(dff, frame=frame_idx, alpha=alpha, iterations=iterations)

        flow[:,:,0] *=data["dff"][:,:,frame_idx]
        flow[:,:,1] *=data["dff"][:,:,frame_idx]

        result = wavefront_detection(dff, frame=frame_idx, flow=flow, x=x, y=y, rect_size=rect_size, sigma=sigma, prominence=prominence)

        frame = result["frame"]
        gradient = result["gradient"]
        corners = result["corners"]
        masked_gradient = result["masked_gradient"]
        profile = result["profile"]
        smoothed_profile = result["smoothed_profile"]
        lim_up = result["lim_up"]
        lim_down = result["lim_down"]

        # Same extrema detection as wavefront analysis
        peaks, _ = find_peaks(smoothed_profile, prominence=prominence, distance=5)
        peaks = np.array([i for i in peaks if smoothed_profile[i] > lim_up])

        troughs, _ = find_peaks(-smoothed_profile, prominence=prominence, distance=5)
        troughs = np.array([i for i in troughs if smoothed_profile[i] < lim_down])

        vmax = np.nanmax(np.abs(gradient))
        polygon = np.vstack([corners, corners[0]])

        axes[0, 3].set_ylabel("Temporal Gradient [intensity/frame]",fontsize=7)


        ### 1. dF/F + optic flow
        axes[row, 0].imshow(frame, cmap="Blues", vmin=0, vmax=1)

        if row in [0, 1]:
            plot_quiver(axes[row, 0], flow, spacing=spacing, scale=scale, color="black")
        elif row == 2:
            plot_quiver(axes[row, 0], flow, spacing=spacing, scale=scale, color="black")
        elif row == 3:
            plot_quiver(axes[row, 0], flow, spacing=spacing, scale=0.01, color="black")

        axes[row, 0].set_ylim(frame.shape[0], 0)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])


        ### 2. Temporal gradient
        im = axes[row, 1].imshow(gradient, cmap="RdBu", vmin=-vmax, vmax=vmax)

        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        ### 3. dF/F + flow + temporal gradient inside rectangle
        axes[row, 2].imshow(frame, cmap="Blues", vmin=0, vmax=1)
        if row in [0, 1]:
            plot_quiver(axes[row, 2], flow, spacing=spacing, scale=scale, color="black")
        elif row == 2:
            plot_quiver(axes[row, 2], flow, spacing=spacing, scale=scale, color="black")
        elif row == 3:
            plot_quiver(axes[row, 2], flow, spacing=spacing, scale=0.01, color="black")

        masked_overlay = np.ma.masked_invalid(masked_gradient)
        axes[row, 2].imshow(masked_overlay, cmap="RdBu", vmin=-vmax, vmax=vmax, alpha=0.9)

        axes[row, 2].plot(polygon[:, 0], polygon[:, 1], color="black", linewidth=0.5)
        axes[row, 2].scatter(x, y, color="darkviolet", s=35, zorder=5)

        axes[row, 2].set_ylim(frame.shape[0], 0)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])


        ### 4. Gradient profile
        axes[row, 3].plot(profile, marker='o', linestyle='-', color='royalblue', markersize=4, alpha=0.8, linewidth=1.5, label='Gradient profile')

        axes[row, 3].plot(smoothed_profile, linestyle='-', color='brown', linewidth=1.5, label='Smoothed profile')

        axes[row, 3].axhline(lim_down, color='black', linestyle='--', linewidth=1)
        axes[row, 3].axhline(lim_up, color='black', linestyle='--', linewidth=1)


        if len(peaks) == 1 and len(troughs) == 1 and troughs[0] < peaks[0]:
            axes[row, 3].scatter(peaks[0], smoothed_profile[peaks[0]], color='green', s=60, label='Maximum', zorder=5)
            axes[row, 3].scatter(troughs[0], smoothed_profile[troughs[0]], color='red', s=60, label='Minimum', zorder=5)

        # Dataset name on left side
        axes[row, 0].set_title(title, loc="left")

        if row in [0, 1]:
            axes[row, 3].set_ylim(-0.25, 0.25)
        elif row == 2:
            axes[row, 3].set_ylim(-0.15, 0.15)
        elif row == 3:
            axes[row, 3].set_ylim(-0.10, 0.10)

    # ---------------------------------------------------------
    # Column titles
    # ---------------------------------------------------------
    axes[0, 0].set_title("dF/F + Optic Flow")
    axes[0, 1].set_title("Temporal Gradient")
    axes[0, 2].set_title("Wavefront Search Region")
    axes[0, 3].set_title("Gradient Profile")

    axes[-1, 3].set_xlabel("Position along search line")

    # One legend instead of 4 repeated legends
    handles, labels = axes[0, 3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()

    return fig, axes

def bresenham_line(x0, y0, x1, y1):

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

def wavefront_detection(dff, frame, flow, x, y, rect_size=(34, 4), sigma=1, prominence=0, boundary_condition=False, brain_mask=None):

    current_frame = dff[:, :, frame]
    gradient_map = dff[:, :, frame + 1] - current_frame

    direction = flow[y, x, :]

    vx, vy = direction
    norm = np.hypot(vx, vy)

    # Same behavior as original implementation
    if norm < 0.001:
        return None

    unit_vx = vx / norm
    unit_vy = vy / norm

    rect_h, rect_w = rect_size
    H, W = current_frame.shape

    # Same rectangle construction
    dx_w = (rect_w / 2) * unit_vy
    dy_w = -(rect_w / 2) * unit_vx

    dx_h = (rect_h / 2) * unit_vx
    dy_h = (rect_h / 2) * unit_vy

    corners = np.array([
        [x - dx_w - dx_h, y - dy_w - dy_h],
        [x + dx_w - dx_h, y + dy_w - dy_h],
        [x + dx_w + dx_h, y + dy_w + dy_h],
        [x - dx_w + dx_h, y - dy_w + dy_h]
    ], dtype=np.float32)

    corners = np.clip(corners, [0, 0], [W - 1, H - 1]).astype(np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.reshape((-1, 1, 2))], 1)

    masked_gradients = gradient_map * mask

    # ---------------------------------------------------------
    # Same optional boundary correction as original
    # ---------------------------------------------------------
    num_zeros = np.sum((mask == 1) & (masked_gradients == 0))

    if num_zeros > 0 and boundary_condition:

        max_extension = int(rect_h / 2)
        step_size = 1

        directions = {
            'backward': (-unit_vx, -unit_vy),
            'forward': (unit_vx, unit_vy),
            'left': (unit_vy, -unit_vx),
            'right': (-unit_vy, unit_vx)
        }

        for step in range(1, max_extension + 1):

            best_num_zeros = num_zeros
            best_corners = None
            best_mask = None
            best_masked_gradients = None

            for dir_name, (dx, dy) in directions.items():

                if dir_name in ["left", "right"] and step > 4:
                    continue

                shift_x = dx * step * step_size
                shift_y = dy * step * step_size

                x_shifted = x + shift_x
                y_shifted = y + shift_y

                test_corners = np.array([
                    [x_shifted - dx_w - dx_h, y_shifted - dy_w - dy_h],
                    [x_shifted + dx_w - dx_h, y_shifted + dy_w - dy_h],
                    [x_shifted + dx_w + dx_h, y_shifted + dy_w + dy_h],
                    [x_shifted - dx_w + dx_h, y_shifted - dy_w + dy_h]
                ], dtype=np.float32)

                test_corners = np.clip(test_corners, [0, 0], [W - 1, H - 1]).astype(np.int32)

                test_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(test_mask, [test_corners.reshape((-1, 1, 2))], 1)

                test_masked_gradients = gradient_map * test_mask
                new_zeros = np.sum((test_mask == 1) & (test_masked_gradients == 0))

                if new_zeros < best_num_zeros:
                    best_num_zeros = new_zeros
                    best_corners = test_corners
                    best_mask = test_mask
                    best_masked_gradients = test_masked_gradients

            if best_num_zeros < num_zeros:
                corners = best_corners
                mask = best_mask
                masked_gradients = best_masked_gradients
                num_zeros = best_num_zeros

            if num_zeros == 0:
                break

    # ---------------------------------------------------------
    # Same profile extraction as original
    # ---------------------------------------------------------
    start_x, start_y = corners[0]
    end_x, end_y = corners[2]

    line_pixels = bresenham_line(start_x, start_y, end_x, end_y)

    profile = np.array([
        gradient_map[yy, xx]
        for xx, yy in line_pixels
        if 0 <= xx < W and 0 <= yy < H
    ])

    profile = profile[profile != 0]

    smoothed_profile = gaussian_filter1d(profile, sigma=sigma)

    # ---------------------------------------------------------
    # Same threshold calculation as original process_data()
    # ---------------------------------------------------------
    grad_t = np.gradient(dff, axis=2)

    if brain_mask is not None and grad_t.shape[:2] == brain_mask.shape:
        masked_grad = grad_t[brain_mask, :]
    else:
        masked_grad = grad_t

    avg_grad = masked_grad.mean()
    std_grad = masked_grad.std()

    lim_up = avg_grad + std_grad
    lim_down = avg_grad - std_grad

    score = find_max_min(profile, sigma=sigma, prominence=prominence, lim_up=lim_up, lim_down=lim_down)

    masked_gradient_nan = np.where(masked_gradients == 0, np.nan, masked_gradients)

    return {
        "frame": current_frame,
        "gradient": gradient_map,
        "flow": flow,
        "direction": direction,
        "corners": corners,
        "masked_gradient": masked_gradient_nan,
        "profile": profile,
        "smoothed_profile": smoothed_profile,
        "lim_up": lim_up,
        "lim_down": lim_down,
        "score": score
    }

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
        dff1 = create_patterns(N=64, M=64, frames=33, pattern='radial', x0=32, y0=32, sd0=2, u=0, v=1, rad_spd=1,rad_width=5)

        title = 'Radial Wave'

        return dff1, title


dff1, _ = data_type('radial')

dff2, _ = data_type('radial')
add_noise(dff2, std=0.03)

dff3  = np.load('Fig 2 Example.npy')

retina_data = np.load('Fig 4 - datasets/retina example.npz')
dff4 = retina_data["data"].astype(np.float32)

datasets = [
    {"dff": dff1, "frame": 19, "x": 45, "y": 46, "title": " "},
    {"dff": dff2, "frame": 19, "x": 45, "y": 46, "title": " "},
    {"dff": dff3, "frame": 20, "x": 21, "y": 62, "title": " "},   ## optional
    {"dff": dff4, "frame": 200, "x": 83, "y": 82, "title": " "}
]

plot_wavefront_detection_4rows(datasets)