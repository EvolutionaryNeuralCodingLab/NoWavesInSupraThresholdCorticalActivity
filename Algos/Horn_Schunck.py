import numpy as np
from scipy.signal import convolve2d , convolve


def horn_schunck(im1, im2, alpha, num_iter):
    def avg(u):
        avg_stencil = np.array([[0, 1/4, 0],
                                [1/4, 0, 1/4],
                                [0, 1/4, 0]])

        return convolve2d(u, avg_stencil, mode='same', boundary='symm')

    def derivative(matrix):
        sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        dx = convolve2d(matrix, sobel_x , mode='same', boundary='symm')
        dy = convolve2d(matrix, sobel_y , mode='same', boundary='symm')

        return dx, dy

    # Initialize the flow vectors to zero
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)



    ## If phae
    #It = np.angle(im2 * np.conj(im1))
    #re = np.real(im1)
    #im = np.imag(im1)
    #Ix_re, Iy_re = derivative(re)
    #Ix_im, Iy_im = derivative(im)
    #Ix = -im * Ix_re + re * Ix_im
    #Iy = -im * Iy_re + re * Iy_im
    #####

    # Compute derivatives of the images using central difference
    Ix , Iy = derivative(im1)
    It = im2 - im1




    Ix_sq = np.square(Ix)
    Iy_sq = np.square(Iy)
    alpha_sq = alpha ** 2

    # Precompute denominator (remains constant across iterations)
    denominator = alpha_sq + Ix_sq + Iy_sq

    convergence = []


    for i in range(num_iter):
        u_avg = avg(u)
        v_avg = avg(v)

        term = (Ix * u_avg + Iy * v_avg + It) / denominator

        # Store old u, v
        u_prev = u.copy()
        v_prev = v.copy()

        # Update flow
        u = u_avg - Ix * term
        v = v_avg - Iy * term

        # Compute flow update magnitude (per pixel L2 norm)
        delta_u = u - u_prev
        delta_v = v - v_prev

        epsilon = 1e-6
        flow_magnitude = np.sqrt(u ** 2 + v ** 2)
        rel_update = np.sqrt(delta_u ** 2 + delta_v ** 2) / (flow_magnitude + epsilon)
        mean_rel_update = rel_update.mean()

        convergence.append(mean_rel_update)  # now relative
    ## IF PhASE:
    #u = np.real(u)
    #v = np.real(v)
    ####

    return np.stack([u, v], axis=2), convergence


def horn_schunck_phase(phi1, phi2, alpha, num_iter):

    def avg(u):
        avg_stencil = np.array([[0, 1/4, 0],
                                [1/4, 0, 1/4],
                                [0, 1/4, 0]])
        return convolve2d(u, avg_stencil, mode='same', boundary='symm')

    def circular_diff(target, reference):
        return (target - reference + np.pi) % (2 * np.pi) - np.pi

    def derivative_circular(phi):
        padded = np.pad(phi, pad_width=1, mode='reflect')

        dx = circular_diff(padded[1:-1, 2:], padded[1:-1, :-2]) / 2.0
        dy = circular_diff(padded[2:, 1:-1], padded[:-2, 1:-1]) / 2.0

        return dx, dy

    u = np.zeros_like(phi1)
    v = np.zeros_like(phi1)

    Ix, Iy = derivative_circular(phi1)
    It = circular_diff(phi2, phi1)

    denominator = alpha**2 + Ix**2 + Iy**2 + 1e-6

    convergence = []

    for _ in range(num_iter):
        u_avg = avg(u)
        v_avg = avg(v)

        term = (Ix * u_avg + Iy * v_avg + It) / denominator

        u_new = u_avg - Ix * term
        v_new = v_avg - Iy * term

        delta = np.sqrt((u_new - u)**2 + (v_new - v)**2)
        mag = np.sqrt(u_new**2 + v_new**2)

        convergence.append(np.mean(delta / (mag + 1e-6)))

        u, v = u_new, v_new

    return np.stack([u, v], axis=2), convergence


def lucas_kanade(prev_frame, next_frame, window_size):
    # Calculate spatial gradients (Sobel filters)
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    I_x = convolve(prev_frame, sobel_x)
    I_y = convolve(prev_frame, sobel_y)
    I_t = next_frame - prev_frame

    half_window = window_size // 2

    u = np.zeros(prev_frame.shape)
    v = np.zeros(prev_frame.shape)

    # Iterate over the image to calculate optical flow for each pixel
    for i in range(half_window, prev_frame.shape[0] - half_window):
        for j in range(half_window, prev_frame.shape[1] - half_window):
            # Extract window around the pixel
            I_x_window = I_x[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()
            I_y_window = I_y[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()
            I_t_window = I_t[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()

            A = np.vstack((I_x_window, I_y_window)).T
            b = -I_t_window

            # Solve the linear system
            if np.linalg.matrix_rank(A) == 2:
                flow = np.linalg.lstsq(A, b, rcond=None)[0]
                u[i, j] = flow[0]
                v[i, j] = flow[1]


    return np.stack([u, v], axis=2)





