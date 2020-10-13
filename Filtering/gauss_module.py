# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2


"""
Convolve function [...]
"""
def convolve_ex(img, kernel):

    # Shapes of Kernel + Image
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = img.shape[0]
    yImgShape = img.shape[1]

    # Shape of Output Convolution
    xOutput = int((xImgShape - xKernShape) + 1)
    yOutput = int((yImgShape - yKernShape) + 1)
    smooth_img = np.zeros((xOutput, yOutput))

    # Iterate through image
    for y in range(img.shape[1] - yKernShape):
        for x in range(img.shape[0] - xKernShape):
            smooth_img[x, y] = (kernel * img[x: x + xKernShape, y: y + yKernShape]).sum()

    return smooth_img

def convolve(img, kernel):

    image_height = img.shape[0]
    image_width = img.shape[1]

    kernel_height = kernel.shape[0]
    kernel_halfh = kernel_height // 2
    kernel_width = kernel.shape[1]
    kernel_halfw = kernel_width // 2

    smooth_img = np.zeros((image_height, image_width))

    # Do convolution
    for x in range(image_width):
        for y in range(image_height):
            # Calculate usable image / kernel range
            x_min = max(0, x - kernel_halfw)
            x_max = min(image_width - 1, x + kernel_halfw)
            y_min = max(0, y - kernel_halfh)
            y_max = min(image_height - 1, y + kernel_halfh)

            # Convolve filter
            value = 0
            total = 0
            for u in range(x_min, x_max + 1):
                for v in range(y_min, y_max + 1):
                    tmp = kernel[v - y + kernel_halfh, u - x + kernel_halfw]
                    value += img[v, u] * tmp
                    total += tmp
            smooth_img[y, x] = value / total

    return smooth_img


"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):

    #...
    x = np.arange(-3 * sigma, 3 * sigma + 1, 1, dtype='int32')
    Gx = np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.))) * (1 / (np.sqrt(2 * np.pi) * sigma))

    return Gx, x



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter_2d(img, sigma):
    kernel = gauss(sigma)[0]
    kernel = np.outer(kernel, kernel)
    kernel = kernel/kernel.sum()
    smooth_img = conv2(img, kernel)

    return smooth_img


def gaussianfilter(img, sigma):
    kernel = gauss(sigma)[0]
    kernel = (kernel/kernel.sum()).reshape(1,kernel.shape[0])

    smooth_img = conv2(img, kernel)
    smooth_img = conv2(smooth_img, kernel.transpose())

    return smooth_img


"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    x = np.arange(-3 * sigma, 3 * sigma + 1, 1, dtype=np.int)
    Dx = - np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.))) * x * (1 / (np.sqrt(2 * np.pi) * np.power(sigma,3.)))

    return Dx, x



def gaussderiv(img, sigma):
    kernel = gaussdx(sigma)[0]
    kernel = (kernel/kernel.sum()).reshape(1,kernel.shape[0])

    imgDx = convolve(img, kernel)
    imgDy = convolve(img, kernel.transpose())

    return imgDx, imgDy
