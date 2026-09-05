import scipy
import numpy as np
from scipy.ndimage import uniform_filter
import cv2 as cv
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy.signal import butter, filtfilt

def bilateral_filter(img, sigma_spatial, sigma_range):
    """
    Bilateral filter to a 2D numpy array.
    """
    return cv.bilateralFilter(img.astype(np.float32), -1, sigma_spatial, sigma_range)

def gaussian_filter(img, sigma):
    """
    Gaussian filter to a 2D numpy array.
    """
    return cv.GaussianBlur(img.astype(np.float32), (0, 0), sigma)

def average_filter(img, kernel_size):
    """
    Average filter to a 2D numpy array.
    """
    return cv.blur(img, (kernel_size, kernel_size))

def normalize_box_blur(img, kernel_size):
    """
    Normalize box blur to a 2D numpy array.
    """
    return cv.boxFilter(img, -1, (kernel_size, kernel_size), normalize=True)



#### Imported modules to main code

def resize(img ,N,M):
    return cv.resize(img, (N,M), interpolation = cv.INTER_AREA)
    #return cv.resize(img, (N,M), interpolation = cv.INTER_LANCZOS4)


def Filter(img,fil,sigma,kernel):
    if fil == 'gaussian':
        return gaussian_filter(img, sigma=sigma)
    if fil == 'bilateral':
        return bilateral_filter(img, sigma_spatial=sigma, sigma_range=7)
    if fil == 'average':
        return average_filter(img, kernel_size=kernel)
    if fil == 'normalize':
        return normalize_box_blur(img, kernel_size=kernel)
    ## Sigma - for gaussian and bilateral
    ## kernel - for average and normalize


def pca(img,n_comp):
    pca = PCA(n_components = n_comp)
    components = pca.fit_transform(img)
    return pca.inverse_transform(components)


def decrease_frame_rate(video_array, original_fps, target_fps):
    # Calculate the frame skip factor
    skip_factor = original_fps / target_fps

    # Calculate the Nyquist frequency based on the target frame rate
    nyquist = 0.5 * target_fps
    cutoff_frequency = target_fps / original_fps

    # Define a low-pass filter
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(2, normal_cutoff, btype='low', analog=False)

    # Apply the filter across the time dimension
    filtered_frames = np.zeros_like(video_array)
    for i in range(video_array.shape[0]):
        for j in range(video_array.shape[1]):
            filtered_frames[i, j, :] = filtfilt(b, a, video_array[i, j, :], axis=0)

    def generate_custom_indices(length):
        indices = []
        current_index = 1
        step = 3
        while current_index < length:
            indices.append(current_index)
            current_index += step
            step = 2 if step == 3 else 3  # Alternate between adding 3 and 2
        return indices

    def reduce_video_frames(filtered_frames, skip_factor):
        if skip_factor % 1 == 0 and skip_factor > 0:
            # Handle the case for integer skip_factor
            reduced_video_array = filtered_frames[:, :, ::int(skip_factor)]
        elif skip_factor == 2.5:
            # Handle the specific case where skip_factor is 2.5 using custom indices
            num_frames = filtered_frames.shape[2]
            custom_indices = generate_custom_indices(num_frames)
            reduced_video_array = np.stack([filtered_frames[:, :, i] for i in custom_indices], axis=2)
        else:
            raise ValueError("Invalid skip_factor. Only positive integers or 2.5 are allowed.")

        return reduced_video_array

    return reduce_video_frames(filtered_frames, skip_factor)


def normalize_data(data):
    non_zero_data = data[data != 0]

    flattened_data = abs(non_zero_data).flatten()

    sorted_data = np.sort(flattened_data)[::-1]  # Sort the flattened data in descending order

    percentile_index = int(len(sorted_data) * 0.001)  # Determine the index corresponding to the x-th percentile

    max_value = sorted_data[percentile_index - 1]  # Retrieve the maximum value with respect to chosen value of the data
    normalized_data = data / max_value
    normalized_data[normalized_data > 1] = 1  ## Make every value > 1 be 1
    return normalized_data


