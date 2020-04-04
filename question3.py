import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import imageio
import math

#helper for gaus function
def gaus_blur(sigma, x, y):
    return np.exp(-1.0 * (x**2 + y**2)/2*sigma**2)/2*sigma**2

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

#NOTE: My ide was complaining about the file you had given us so I just copied the function here!
def intensityscale(raw_img):
    #scale an image's intensity from [min, max] to [0, 1]
    v_min, v_max = raw_img.min(), raw_img.max()
    scaled_im = (raw_img * 1.0 - v_min) / (v_max - v_min)
    # keep the mean to be 0.5.
    meangray = np.mean(scaled_im)
    scaled_im = scaled_im - meangray + 0.5
    
    # clip to [0, 1]
    scaled_im = np.clip(scaled_im, 0, 1)   
    return scaled_im

def FFT_Blur(image, kernel, fft_blur = None):
    
    extra_blur = np.ones((5, 5))/25
    kernel_highpass = 1 - kernel
    #compute DFT mask
    faceDFT_1 = np.fft.fftshift(np.fft.rfft2(image))
    #convolve and reverse FT using normal kernel
    face = np.fft.irfft2(np.fft.ifftshift(signal.convolve2d(faceDFT_1, kernel, 'same')))

    #display FFT mask
    plt.imshow(face, cmap="gray")
    plt.title("FFT of Image ({0}x{1} Low pass)".format(*kernel.shape))
    plt.show()

    #extract image and display, by adding the mask those regions "intensify" so once its scaled they
    #are easier to see
    extract_f1 = signal.convolve2d(image + face, kernel, 'same')
    # extract_f1 = signal.convolve2d(extract_f1, np.ones((3,3))/9, 'same')

    #subtract the mask to reduce the expression of those regions
    extract_f2 = signal.convolve2d(image - face , kernel_highpass, 'same')
    # extract_f2 = signal.convolve2d(extract_f2, np.ones((3,3))/9, 'same')
    #display extracted images
    plt.subplot(1, 2, 1)
    plt.imshow(intensityscale(extract_f1), cmap="gray")
    plt.title("Extracted Image 1 ")
    plt.subplot(1, 2, 2)
    plt.imshow(intensityscale(extract_f2), cmap="gray")
    plt.title("Extracted Image 2 ")
    plt.show()

#load image and scale
image = imageio.imread('einsteinandwho.png')
#show original image
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.show()

sigmaVals = [0.5, 1, 3/2 , 2, 3]
#create several gaus kernels
kernels = [create_gaus_kernel(x) for x in sigmaVals]

for x in range(len(kernels)):
    FFT_Blur(image, kernels[x])


