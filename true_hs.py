
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve as convolve2d
import os
from argparse import ArgumentParser
from scipy.signal import convolve2d

"""
see readme for running instructions
"""


def show_image(name, image):
    if image is None:
        return

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#compute magnitude in each 8 pixels. return magnitude average
def get_magnitude(u, v):
    scale = 3
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg



def draw_quiver(u,v,beforeImg):
    scale = 3
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap = 'gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 16):
        for j in range(0, u.shape[1],16):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            # magnitude = (dx**2 + dy**2)**0.5
            # #draw only significant changes
            # if magnitude > magnitudeAvg:
            #     ax.arrow(j,i, dx, dy, color = 'red')
            ax.arrow(j,i, dx, dy, color = 'red')

    plt.draw()
    plt.savefig("true_hs.png")
    plt.show()






HSKERN = np.array(
    [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], float
)

kernelX = np.array([[-1, 1], [-1, 1]]) * 0.25  # kernel for computing d/dx

kernelY = np.array([[-1, -1], [1, 1]]) * 0.25  # kernel for computing d/dy

kernelT = np.ones((2, 2)) * 0.25


def HornSchunck(
    im1: np.ndarray,
    im2: np.ndarray,
    *,
    alpha: float = 0.001,
    Niter: int = 8,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------

    im1: numpy.ndarray
        image at t=0
    im2: numpy.ndarray
        image at t=1
    alpha: float
        regularization constant
    Niter: int
        number of iteration
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)
    vInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Iteration to reduce error
    for _ in range(Niter):
        # %% Compute local averages of the flow vectors
        uAvg = convolve2d(U, HSKERN, "same")
        vAvg = convolve2d(V, HSKERN, "same")
        # %% common part of update step
        der = (fx * uAvg + fy * vAvg + ft) / (alpha ** 2 + fx ** 2 + fy ** 2)
        # %% iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V


def computeDerivatives(
    im1: np.ndarray, im2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    fx = convolve2d(im1, kernelX, "same") + convolve2d(im2, kernelX, "same")
    fy = convolve2d(im1, kernelY, "same") + convolve2d(im2, kernelY, "same")

    # ft = im2 - im1
    ft = convolve2d(im1, kernelT, "same") + convolve2d(im2, -kernelT, "same")

    return fx, fy, ft


if __name__ == '__main__':
    parser = ArgumentParser(description = 'Horn Schunck program')
    parser.add_argument('img1', type = str, help = 'First image name (include format)')
    parser.add_argument('img2', type = str, help='Second image name (include format)')
    args = parser.parse_args()

    im1 = cv2.imread(args.img1, 0).astype(float)
    im2 = cv2.imread(args.img2, 0).astype(float)

    U, V = HornSchunck(im1, im2, alpha=1.0, Niter=100)
    draw_quiver(U,V, im1)




