import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import imageio
import math

#controller for various displays/tests
#setting to false will result in that section not executing
OG_IMG = False
BOX_BLUR = True
VIS_GAUS = False
GAUS_BLUR = False
FFT_BLUR = True

#function for 2D gaus value
def gaus_blur(sigma, x, y):
    return np.exp(-1.0 * ((x**2 + y**2)/(2 * sigma**2))) / (2*math.pi*sigma**2)

#grayscale images (from matlab)
def grayScale(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

#helper function for making a gaussian kernel
def create_gaus_kernel(std1):
    kernel_size = int(4*std1) + 1
    kernel = np.zeros((kernel_size, kernel_size))
    centerRows = kernel_size//2
    centerCols = kernel_size//2
    for i in range(kernel_size):
        x = i - centerRows
        for j in range(kernel_size):
            y = j - centerCols
            kernel[i, j] = gaus_blur(std1, x, y)
    
    kernel *= 1.0/np.sum(kernel)

    return kernel

def filter(image1, image2, kernel, kernel_name):
    a_blurred = signal.convolve2d(image1, kernel, 'same')
    b_blurred = signal.convolve2d(image2, kernel, 'same')
    output = b_blurred + (image1 - a_blurred)
    #display the blurred images
    plt.subplot(1, 2, 1)
    plt.title("Good ol' Abe Blurred "+kernel_name)
    plt.imshow(a_blurred, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Tiger Blurred "+kernel_name)
    plt.imshow(b_blurred, cmap="gray")
    plt.show()
    #display the hybrid image
    plt.title("Hybrid Image "+kernel_name)
    plt.imshow(output, cmap="gray")
    plt.show()

#load in both images
image1 = grayScale(imageio.imread('abe.jpg'))
image2 = grayScale(imageio.imread('tiger.jpg'))

if OG_IMG:
    #create subplots displaying original images
    plt.suptitle("Original Images")
    plt.subplot(1, 2, 1)
    plt.title("Good ol' Abe")
    plt.imshow(image1, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Tiger")
    plt.imshow(image2, cmap="gray")
    plt.show()

if BOX_BLUR:
    #Box kernel window sizes
    box_kernel_sizes = [3, 5, 7, 9, 11]
    #create box kernels
    box_kernels = [np.ones((x, x))/x**2 for x in box_kernel_sizes]

    #filter images using box kernels and display them (low pass only)
    for i in range(len(box_kernels)): 
        filter(image1, image2, box_kernels[i], "\n{0}x{0} Box Kernel".format(box_kernel_sizes[i]))


#create Gaus Kernels for 3x3, 5x5, 7x7, 9x9, 
sigmaVals = [0.5, 1, 3/2, 2, 3]
Gaussian_kernels = [create_gaus_kernel(x) for x in sigmaVals]
# Visualize gaus filters
if VIS_GAUS:
    plt.suptitle("Gaussian Filters and Inverses \nSigma = (0.5, 1, 2, 3, 7)")
    for x in range(len(Gaussian_kernels)):
        plt.subplot(5, 2, 2*x + 1)
        plt.imshow(Gaussian_kernels[x])
        plt.subplot(5, 2, 2*x + 2)
        plt.imshow(1 - Gaussian_kernels[x])
    plt.show()

if GAUS_BLUR:
    #filter images using gaussian kernels and display them (low pass only)
    for i in range(len(sigmaVals)): 
        filter(image1, image2, Gaussian_kernels[i], "\n{0}x{0} Gaussian Kernel".format(int(4*sigmaVals[i]) + 1))


if FFT_BLUR:
    #FFT blurring with high/low pass
    #Compute the discrete fft of the images and shift the result so its centered
    faceDFT_1 = np.fft.fftshift(np.fft.rfft2(image1))
    faceDFT_2 = np.fft.fftshift(np.fft.rfft2(image2))
    sigma = 0
    for z in Gaussian_kernels:
        a_blurred = np.fft.irfft2(np.fft.ifftshift(signal.convolve2d(faceDFT_1, z, 'same')))
        b_blurred = np.fft.irfft2(np.fft.ifftshift(signal.convolve2d(faceDFT_2, 1-z, 'same')))

        plt.suptitle("{0}x{1} Gaussian Blur".format(*z.shape))
        plt.subplot(1, 2 ,1)
        plt.imshow(a_blurred, cmap="gray")
        plt.title("FFT of Face 1 ")

        plt.subplot(1,2,2)
        plt.imshow(b_blurred, cmap="gray")
        plt.title("FFT of Face 2")
        plt.show()

        hybrid = b_blurred + (image1 - a_blurred)
        plt.imshow(hybrid, cmap="gray")
        plt.title("Hybrid using FFT ({0}x{1} Gaussian)".format(*z.shape))
        plt.show()
        sigma += 1