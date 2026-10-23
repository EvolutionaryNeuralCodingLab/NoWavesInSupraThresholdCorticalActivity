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
        positions = [0.0, 0.5, 1]

        self.color_map = LinearSegmentedColormap.from_list("AbyssBlue", list(zip(positions, colors)))

        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.iter = iterations
        self.lim = lim

    def plot_frames(self, space, scale):

        frame_indices = [3, 17, 21, 27, 35, 43, 52, 69]

        n_cols = len(frame_indices)

        # Wider and shorter figure for 8 columns × 3 rows
        figsize_cm = (24, 9)
        figsize_in = tuple(x / 2.54 for x in figsize_cm)

        fig = plt.figure(figsize=figsize_in, dpi=350)

        gs = gridspec.GridSpec(
            3, n_cols,
            height_ratios=[1, 1, 1],
            hspace=0.15,
            wspace=0.03,
            left=0.12,
            right=0.99,
            top=0.92,
            bottom=0.07
        )

        calcium_axes = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
        flow_axes = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]
        momentum_axes = [fig.add_subplot(gs[2, i]) for i in range(n_cols)]

        for i, frame_idx in enumerate(frame_indices):
            # Copy arrays to avoid modifying original data
            flow = self.flows[frame_idx].copy()
            flow[:, :, 0] *= brain_mask
            flow[:, :, 1] *= brain_mask

            momentum = self.phase_space[:, :, frame_idx, :].copy()
            momentum[:, :, 0] *= brain_mask
            momentum[:, :, 1] *= brain_mask

            # ---------------------------------------
            # ROW 1: CALCIUM SIGNAL
            # ---------------------------------------
            ax = calcium_axes[i]
            ax.imshow( self.dff[:, :, frame_idx], cmap=self.color_map, vmin=0, vmax=1 )
            ax.set_aspect('equal')
            ax.axis('off')


            # ---------------------------------------
            # ROW 2: OPTIC FLOW
            # ---------------------------------------
            ax = flow_axes[i]
            plot_quiver( ax, flow, spacing=space, scale=scale, color='black', width=0.006)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.axis('off')

            # ---------------------------------------
            # ROW 3: MOMENTUM
            # ---------------------------------------
            ax = momentum_axes[i]
            plot_quiver( ax, momentum, spacing=space, scale=scale * 0.7, color='black', width=0.006 )
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.axis('off')

        # ---------------------------------------
        # ROW LABELS
        # ---------------------------------------
        fig.text( 0.015, 0.78, r"Calcium signal $I$", ha='left', va='center', fontsize=10, fontname='Arial' )
        fig.text( 0.015, 0.49, r"Optic flow $\vec{V}$", ha='left', va='center', fontsize=10, fontname='Arial' )
        fig.text( 0.015, 0.20, r"Momentum $\vec{P}$", ha='left', va='center', fontsize=10, fontname='Arial' )

        plt.savefig( 'Sup fig 1.pdf', bbox_inches='tight' )
        plt.show()


brain_mask = np.load('brain_mask.npy')


brain_mask = brain_mask


### MAIN EXAMPLE FIG 2
dff1 = np.load('Fig 2 Example.npy')


data = PreDataProcessing(dff1)
data.resize(128,128)
data = FlowAnalyze(data)

data.horn_schunck_flow(alpha=alpha, num_iter=iterations, phase=False)
display = Display(data)
display.title = 'Fig 2 example'
display.plot_frames( space = 6 , scale = 0.04)

