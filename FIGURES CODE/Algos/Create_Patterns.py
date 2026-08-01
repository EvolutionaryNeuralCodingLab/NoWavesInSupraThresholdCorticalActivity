import numpy as np


## N,M - size of  frames
## frames - # of frames
## pattern - 'plane' , 'radial' , 'cont'
## x0,y0 - origin
## (u,v) - (x,y) velocities [px/s]
## sdi - standard deviation gaussian i
## t0_i - temporal peak of gaussian i
## sdTi - temporal standard deviation gaussian i




def create_patterns(N, M, frames, pattern, x0, y0, sd0, u, v, rad_spd , rad_width):
    # Create 2D grid of coordinates
    x, y = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))

    ## Create a numpy array of size N x M x T
    dff = np.zeros((N , M , frames))  ## (y,x,t)

    def gaussian(x, mean, sd):
        gaussian_values = np.exp(-((x - mean) ** 2) / (2 * (sd ** 2)))
        return gaussian_values

    if pattern == 'plane':     #### Needs work

        def gaussian_3d(x, y, z, A, mean, sigma):
            return A * np.exp(-((y - mean[0]) ** 2 / (2 * sigma[0] ** 2)))

        # Set parameters
        A = 1
        mean = [0, 128, 0]
        sigma = [rad_width,1.0, 1.0]  # Larger sigma_y for spreading along y-axis

        # Generate 2D grid
        x = np.linspace(0, M, M)
        y = np.linspace(0, N, N)
        x, y = np.meshgrid(x, y)

        # Set a fixed z value for projection (e.g., the mean of the Gaussian)
        z_projection = mean[2]

        dff1 = np.zeros((N, M, frames))


        for i in range(frames):
            dff1[:, :, i]= gaussian_3d(x  , y- y0 + v*i, z_projection, A, mean, sigma)

        return dff1

    if pattern == 'radial':
        dff = np.zeros((N, M, frames))
        X, Y = np.meshgrid(np.arange(M), np.arange(N))
        R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)

        for t in range(frames):
            dff[:, :, t] = gaussian(R, rad_spd * t, rad_width) #* np.exp(-0.02 * t)
        return dff

    if pattern == 'cont':
        dff = np.zeros((N, M, frames))

        for t in range(frames):
            t0 = 26  # Center of the temporal envelope
            sd_t = 7  # Width of the temporal envelope
            envelope = 1#np.exp(-(t - t0) ** 2 / (2 * sd_t ** 2))
            cont = envelope* np.exp(-((x - x0 - u * t) ** 2 + (y - y0 - v * t) ** 2) / (2 * (sd0 ** 2)))
            dff[:, :, t] = cont
        return dff


def create_gaussians(N, M, frames, num_gaus, x0, y0, sd0, t0_0, sdT0,
                     x1=None, y1=None, sd1=None, t0_1=None, sdT1=None,
                     x2=None, y2=None, sd2=None, t0_2=None, sdT2=None):
    # Create 2D grid of coordinates
    x, y = np.meshgrid(np.linspace(0, N-1, N),np.linspace(0, M-1, M))

    ## Create a numpy array of size N x M x T
    dff = np.zeros((M , N , frames))  ## (y,x,t)

    if num_gaus == 1:
        for t in range(frames):
            gaussian1 = np.exp(- (t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                        * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

            dff[:, :, t] = gaussian1
        return dff, (N, M, frames, num_gaus, x0, y0, sd0, t0_0, sdT0)

    if num_gaus == 2:
        for t in range(frames):
            gaussian1 = np.exp(- (t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                        * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

            gaussian2 = np.exp(- (t - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                        * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))


            dff[:, ::-1, t] = gaussian1 + gaussian2

        return dff, (N, M, frames, num_gaus, x0, y0, sd0, t0_0, sdT0,x1, y1, sd1, t0_1, sdT1)

    if num_gaus == 3:
        for t in range(frames):
            gaussian1 = np.exp(- (t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                        * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

            gaussian2 = np.exp(- (t - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                        * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))

            gaussian3 = np.exp(- (t - t0_2) ** 2 / (2 * sdT2 ** 2)) \
                        * np.exp(-((x - x2) ** 2 + (y - y2) ** 2) / (2 * (sd2 ** 2)))

            dff[:, ::-1, t] = gaussian1 + gaussian2 + gaussian3

        return dff, (N, M, frames, num_gaus, x0, y0, sd0, t0_0, sdT0, x1, y1, sd1, t0_1, sdT1, x2, y2, sd2, t0_2, sdT2)


def create_gaussians_moving(N, M, frames, num_gaus, x0, y0, sd0, t0_0, sdT0,
                     x1=None, y1=None, sd1=None, t0_1=None, sdT1=None,
                     x2=None, y2=None, sd2=None, t0_2=None, sdT2=None):
    # Create 2D grid of coordinates
    x, y = np.meshgrid(np.linspace(0, N-1, N),np.linspace(0, M-1, M))

    ## Create a numpy array of size N x M x T
    dff = np.zeros((M , N , frames))  ## (y,x,t)

    if num_gaus == 1:
        for t in range(frames):
            gaussian1 = np.exp(- (t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                        * np.exp(-((x - x0) ** 2 + (y - y0 + 1*t) ** 2) / (2 * (sd0 ** 2)))


            dff[:, :, t] = gaussian1
        return dff, (N, M, frames, num_gaus, x0, y0, sd0, t0_0, sdT0)

    if num_gaus == 2:
        for t in range(frames):
            gaussian1 = np.exp(- (t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                        * np.exp(-((x - x0 - t) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

            gaussian2 = np.exp(- (t - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                        * np.exp(-((x - x1 - t) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))




            dff[:, :, t] = gaussian1 + gaussian2

        return dff, (N, M, frames, num_gaus, x0, y0, sd0, t0_0, sdT0,x1, y1, sd1, t0_1, sdT1)

    if num_gaus == 3:
        for t in range(frames):
            gaussian1 = np.exp(- (t - t0_0) ** 2 / (2 * sdT0 ** 2)) \
                        * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sd0 ** 2)))

            gaussian2 = np.exp(- (t - t0_1) ** 2 / (2 * sdT1 ** 2)) \
                        * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2 * (sd1 ** 2)))

            gaussian3 = np.exp(- (t - t0_2) ** 2 / (2 * sdT2 ** 2)) \
                        * np.exp(-((x - x2) ** 2 + (y - y2) ** 2) / (2 * (sd2 ** 2)))

            dff[:, :, t] = gaussian1 + gaussian2 + gaussian3

        return dff, (N, M, frames, num_gaus, x0, y0, sd0, t0_0, sdT0, x1, y1, sd1, t0_1, sdT1, x2, y2, sd2, t0_2, sdT2)
