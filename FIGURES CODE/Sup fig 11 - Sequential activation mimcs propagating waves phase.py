import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

# =============================================================================
# 1. PARAMETERS
# =============================================================================
Nx, Nt = 201, 201
x = np.linspace(-10, 10, Nx)
t = np.linspace(-10, 10, Nt)
X, T = np.meshgrid(x, t, indexing='ij')

x0, t0 = 1, 1
sigma_x1, sigma_x2 = 1.15, 1
sigma_t1, sigma_t2 = 0.9, 0.9


# =============================================================================
# 2. MANIFOLD GENERATION & HILBERT TRANSFORM
# =============================================================================
A1 = np.exp(-(X - x0)**2 / (2 * sigma_x1**2))
g1 = np.exp(-(T - t0)**2 / (2 * sigma_t1**2))
V1 = A1 * g1

A2 = np.exp(-(X + x0)**2 / (2 * sigma_x2**2))
g2 = np.exp(-(T + t0)**2 / (2 * sigma_t2**2))
V2 = A2 * g2

V = V1 + V2
V_hilbert = hilbert(V, axis=1).imag
phi = np.arctan2(V_hilbert, V)

# =============================================================================
# 3. NUMERICAL & ANALYTICAL DERIVATIVES
# =============================================================================
dx = x[1] - x[0]

# --- Numerical Path ---
phi_unwrapped_x = np.unwrap(phi, axis=0)
phi_x_numerical = np.gradient(phi_unwrapped_x, dx, axis=0)

# --- Analytical Path (Your Formula) ---
# 1. Compute 1D temporal hilbert transforms along the true time vectors
H_g1_1D = hilbert(g1[0, :]).imag
H_g2_1D = hilbert(g2[0, :]).imag

# 2. Broadcast them up to match the 2D meshgrid structure
H_g1 = np.tile(H_g1_1D, (Nx, 1))
H_g2 = np.tile(H_g2_1D, (Nx, 1))

# 3. Construct your components: spatial bracket, temporal bracket, and overlap
spatial_bracket = ((X + x0) / (sigma_x2**2)) - ((X - x0) / (sigma_x1**2))
temporal_bracket = H_g1 * g2 - g1 * H_g2
overlap_numerator = spatial_bracket * (A1 * A2) * temporal_bracket

# 4. Assemble: phi_x = Numerator / (V^2 + H[V]^2)
# Adding a minor regularizer (1e-15) prevents true division-by-zero crashes at dead pixels
phi_x_analytical = overlap_numerator / (V**2 + V_hilbert**2 + 1e-15)

# =============================================================================
# 4. THEORETICAL PREDICTION (Zero-Crossing Axis)
# =============================================================================
if sigma_x1 != sigma_x2:
    x_zero_theoretical = x0 * (sigma_x1**2 + sigma_x2**2) / (sigma_x2**2 - sigma_x1**2)
else:
    x_zero_theoretical = None

t_idx = Nt // 2


# =============================================================================
# 5. VISUALIZATION
# =============================================================================

figsize_cm = (18, 18)
figsize_in = tuple(x / 2.54 for x in figsize_cm)

fig, axs = plt.subplots(3, 2, figsize=figsize_in)
plot_extent = [x[0], x[-1], t[0], t[-1]]

chosen_t = t[t_idx] # The specific time being sliced


def add_matching_colorbar(fig, ax, im, size="4%", pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    return fig.colorbar(im, cax=cax)


colors = [
    (1.0, 1.0, 1.0),  # white
    (0.5, 0.7, 0.8),  # pale blue
    (0.1, 0.2, 0.6),  # navy blue
]
positions = [0.0, 0.5, 1]  # white at 0, pale blue at 0.2, navy at 0.5 (clipped early)
intensity_cmap = LinearSegmentedColormap.from_list("AbyssBlue", list(zip(positions, colors)))


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


phase_cmap = phase_cmap()

param_text = (
    r"$\sigma_{x1} = " + f"{sigma_x1:.1f}" + r",\ \sigma_{x2} = " + f"{sigma_x2:.1f}$" + "\n"
    r"$\sigma_{t1} = " + f"{sigma_t1:.1f}" + r",\ \sigma_{t2} = " + f"{sigma_t2:.1f}$"
)

# Plot A: V(x, t) Space-Time Manifold
imA = axs[0, 0].imshow(V.T, extent=plot_extent, origin='lower', aspect='auto', cmap=intensity_cmap,vmin=0,vmax=1)
#axs[0, 0].set_title(r'Signal $V(x,t)$',fontsize=10)
axs[0, 0].set_xlabel('Space [AU]')
axs[0, 0].set_ylabel('Time [AU]')
cbar=add_matching_colorbar(fig, axs[0, 0], imA)
cbar.set_ticks([0, 1])


# --- NEW: Draw the parameter configuration legend on Plot A ---
# Uses axes fractional coordinates (transform=axs[0,0].transAxes) so it sits safely in the top-left corner
#axs[0, 0].text(0.04, 0.96, param_text, transform=axs[0, 0].transAxes,
#               fontsize=11, verticalalignment='top', horizontalalignment='left',
#               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

# Plot B: Raw Instantaneous Phase
imB = axs[0, 1].imshow(phi.T, extent=plot_extent, origin='lower', aspect='auto', cmap=phase_cmap,    vmin=-np.pi,vmax=np.pi)
#axs[0, 1].set_title(r'Phase $\phi(x,t)$',fontsize=10)
axs[0, 1].set_xlabel('Space [AU]')
axs[0, 1].set_ylabel('Time [AU]')
cbar = add_matching_colorbar(fig, axs[0, 1], imB)
cbar.set_ticks([-np.pi, 0, np.pi])
cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])

# Plot C: Analytical Phase Gradient Map (Your Formula)
imC = axs[1, 0].imshow(phi_x_analytical.T, extent=plot_extent, origin='lower', aspect='auto', cmap='BrBG', vmin=-np.pi/2, vmax=np.pi/2)

# Add the horizontal indicator line for the cross-section slice
axs[1, 0].axhline(chosen_t, color='crimson', linestyle='--', lw=1.5,label=f'Cross-Section')

#if x_zero_theoretical and x[0] < x_zero_theoretical < x[-1]:
#    axs[1, 0].axvline(x_zero_theoretical, color='black', linestyle='--', label=f'Theoretical Zero Node ({x_zero_theoretical:.2f})')

axs[1, 0].legend(loc='upper right',fontsize=8)
#axs[1, 0].set_title(r'Analytical Phase Gradient $\frac{\partial \phi}{\partial x}$',fontsize=10)
axs[1, 0].set_xlabel('Space [AU]')
axs[1, 0].set_ylabel('Time [AU]')
cbar = add_matching_colorbar(fig, axs[1, 0], imC)
cbar.set_ticks([-np.pi/2, 0, np.pi/2])
cbar.set_ticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])

# Plot D: Cross-Section Slice Comparison
axs[1, 1].plot(x, phi_x_numerical[:, t_idx], color='crimson', lw=2.5, label=r'Numerical $\phi_x$')
axs[1, 1].plot(x, phi_x_analytical[:, t_idx], color='black', linestyle='--', lw=2, label=r'Analytical $\phi_x$')
axs[1, 1].axhline(0, color='gray', alpha=0.7, linestyle='-')
#if x_zero_theoretical and x[0] < x_zero_theoretical < x[-1]:
#    axs[1, 1].axvline(x_zero_theoretical, color='darkgreen', linestyle=':', lw=2, label=f'Theoretical Zero Line')
#axs[1, 1].set_title(r'Gradient Profile Cross-Section $\phi_x(x, t_{mid})$',fontsize=10)
axs[1, 1].set_xlabel('Space [AU]')
axs[1, 1].set_ylabel(r'$\frac{\partial \phi}{\partial x}$',fontsize = 12)
axs[1, 1].legend(fontsize=8)
axs[1, 1].grid(True, alpha=0.3)

# Apply your custom zoom constraints uniformly across subplots
xlim_bounds =[-3,3]# [-x0 - 2 * max(sigma_x1, sigma_x2), x0 + 2 * max(sigma_x1, sigma_x2)]
ylim_bounds =[-3,3]# [-t0 - 2 * max(sigma_t1, sigma_t2), t0 + 2 * max(sigma_t1, sigma_t2)]



# =============================================================================
# Plot E-F: Phase latency comparison
# =============================================================================

def phase_latency_1d(phi, amp, t, phi0=0, amp_frac=0.01):
    """
    Compute phase-crossing latency:
        tau(x) = time when phi(x,t) = phi0 mod 2pi
    """
    Nx, Nt = phi.shape
    phi_u = np.unwrap(phi, axis=1)

    tau = np.full(Nx, np.nan)
    amp_thresh = amp_frac * np.nanmax(amp)

    for i in range(Nx):
        p = phi_u[i, :]
        a = amp[i, :]

        k_min = int(np.floor((np.nanmin(p) - phi0) / (2 * np.pi)))
        k_max = int(np.ceil((np.nanmax(p) - phi0) / (2 * np.pi)))

        crossings = []

        for kk in range(k_min, k_max + 1):
            target = phi0 + 2 * np.pi * kk
            y = p - target

            crossing_idx = np.where(np.diff(np.signbit(y)))[0]

            for j in crossing_idx:
                if a[j] < amp_thresh and a[j + 1] < amp_thresh:
                    continue

                denom = y[j + 1] - y[j]
                if np.abs(denom) < 1e-12:
                    continue

                tau_cross = t[j] - y[j] * (t[j + 1] - t[j]) / denom
                crossings.append(tau_cross)

        if crossings:
            crossings = np.array(crossings)
            tau[i] = crossings[np.argmin(np.abs(crossings))]

    return tau


def add_center_rectangle(ax, width=1, height=1, edgecolor="black"):
    rect = Rectangle(
        (-width / 2, -height / 2),   # bottom-left corner
        width,
        height,
        fill=False,
        edgecolor=edgecolor,
        linewidth=1.5,
        linestyle="--"
    )
    ax.add_patch(rect)

# Use the same x, t, X, T grid as the upper plots


# -----------------------------
# Two sequential Gaussian events
# -----------------------------
V_two_latency = V1 + V2

Z_two = hilbert(V_two_latency, axis=1)
phi_two = np.angle(Z_two)
amp_two = np.abs(Z_two)

tau_two = phase_latency_1d(phi_two, amp_two, t)

imF = axs[2, 0].imshow(
    V_two_latency.T,
    extent=plot_extent,
    origin="lower",
    aspect="auto",
    cmap=intensity_cmap,
    vmin=0,
    vmax=1
)

axs[2, 0].plot(
    x,
    tau_two,
    color="orange",
    linewidth=2.5,
    label=" Phase\nlatency"
)



#axs[2, 0].set_title("Two sequential Gaussian events", fontsize=10)
axs[2, 0].set_xlabel("Space [AU]")
axs[2, 0].set_ylabel("Time [AU]")
axs[2, 0].legend(loc="upper left",fontsize=8)
cbar=add_matching_colorbar(fig, axs[2, 0], imF)
cbar.set_ticks([0, 1])


# -----------------------------
# Rise -> constant-amplitude propagation -> decay at end
# -----------------------------
sigma_wave = 0.8

v = 0.4

x_start = -0.5
x_end = 0.5

t_peak = -1.5       # time when Gaussian reaches full amplitude and starts moving
tau_rise = 0.4
tau_decay = 0.4

# time when the moving peak reaches the end
t_end = t_peak + (x_end - x_start) / v

# center: fixed before t_peak, moves until x_end, then stays fixed
x_center = np.piecewise(
    t,[t < t_peak, (t >= t_peak) & (t <= t_end), t > t_end],[x_start,lambda tt: x_start + v * (tt - t_peak),x_end])

# amplitude: rises before t_peak, stays 1 during propagation, decays after t_end
envelope = np.piecewise(
    t,
    [t < t_peak, (t >= t_peak) & (t <= t_end), t > t_end],
    [lambda tt: np.exp(-((tt - t_peak) ** 2) / (2 * tau_rise ** 2)),1.0,lambda tt: np.exp(-((tt - t_end) ** 2) / (2 * tau_decay ** 2))]
)

V_wave = envelope[None, :] * np.exp(-((X - x_center[None, :]) ** 2) / (2 * sigma_wave ** 2))

Z_wave = hilbert(V_wave, axis=1)
phi_wave = np.angle(Z_wave)
amp_wave = np.abs(Z_wave)

tau_wave = phase_latency_1d(phi_wave, amp_wave, t)





imE = axs[2, 1].imshow(
    V_wave.T,
    extent=plot_extent,
    origin="lower",
    aspect="auto",
    cmap=intensity_cmap,
    vmin=0,
    vmax=1
)

axs[2, 1].plot(
    x,
    tau_wave,
    color="orange",
    linewidth=2.5,
    label=" Phase\nlatency"
)

#axs[2, 1].set_title("True traveling Gaussian pulse", fontsize=10)
axs[2, 1].set_xlabel("Space [AU]")
axs[2, 1].set_ylabel("Time [AU]")
axs[2, 1].legend(loc="upper left",fontsize=8)
cbar=add_matching_colorbar(fig, axs[2, 1], imE)
cbar.set_ticks([0, 1])


#add_center_rectangle(axs[2, 0])
#add_center_rectangle(axs[2, 1])


axs[0,0].set_xticks([-3,0,3])
axs[0,0].set_yticks([-3,0,3])
axs[0,1].set_xticks([-3,0,3])
axs[0,1].set_yticks([-3,0,3])
axs[1,0].set_xticks([-3,0,3])
axs[1,0].set_yticks([-3,0,3])
axs[1,1].set_xticks([-3,0,3])
axs[1,1].set_yticks([-3,0,3])
axs[2,0].set_xticks([-3,0,3])
axs[2,0].set_yticks([-3,0,3])
axs[2,1].set_xticks([-3,0,3])
axs[2,1].set_yticks([-3,0,3])

for ax in axs.flat:
    ax.set_xlim(xlim_bounds)
    ax.set_ylim(ylim_bounds)

axs[1,1].set_ylim([-1.5,1.5])
axs[1,1].set_box_aspect(1)   # square axes box
axs[1,1].set_aspect('auto')

axs[0,0].tick_params(axis='x', labelsize=8)
axs[0,0].tick_params(axis='y', labelsize=8)
axs[0,1].tick_params(axis='x', labelsize=8)
axs[0,1].tick_params(axis='y', labelsize=8)
axs[1,0].tick_params(axis='x', labelsize=8)
axs[1,0].tick_params(axis='y', labelsize=8)
axs[1,1].tick_params(axis='x', labelsize=8)
axs[1,1].tick_params(axis='y', labelsize=8)
axs[2,0].tick_params(axis='x', labelsize=8)
axs[2,0].tick_params(axis='y', labelsize=8)
axs[2,1].tick_params(axis='x', labelsize=8)
axs[2,1].tick_params(axis='y', labelsize=8)

axs[0,0].set_aspect('equal', adjustable='box')
axs[0,1].set_aspect('equal', adjustable='box')
axs[1,0].set_aspect('equal', adjustable='box')
#axs[1,1].set_aspect('equal', adjustable='box')
axs[2,0].set_aspect('equal', adjustable='box')
axs[2,1].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig("Phase analysis.svg", bbox_inches='tight', transparent=False, dpi=300)

plt.show()

