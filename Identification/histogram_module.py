import numpy as np
from numpy import histogram as hist

# Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0, filteringpath)
import gauss_module
# needed for convolutions
from scipy.signal import convolve2d as conv2


#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    bins = np.linspace(0, 255, num=num_bins + 1)
    hists = np.zeros(shape=(num_bins), dtype=np.int)
    img_gray_flat = img_gray.flatten()
    for i in range(len(bins) - 1):
        hists[i] = sum(img_gray_flat < bins[i + 1]) - sum(hists)
    hists = hists / sum(hists)

    return hists, bins


#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    img_color_double_flat = np.reshape(img_color_double, (
        img_color_double.shape[0] * img_color_double.shape[1], img_color_double.shape[2]))
    bins = np.linspace(0, 255, num=num_bins + 1)

    # Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins), dtype=np.int)

    # Loop for each pixel i in the image
    for i_pixel in range(img_color_double.shape[0] * img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        color = img_color_double_flat[i_pixel]
        coordinates = np.zeros(3, dtype=np.int)

        for i_color in range(color.shape[0]):
            single_color = color[i_color]
            c = 0
            while c < num_bins and single_color >= bins[c + 1]:
                c += 1
            coordinates[i_color] = c

        hists[coordinates[0], coordinates[1], coordinates[2]] += 1

    # Normalize the histogram such that its integral (sum) is equal 1
    hists = hists / sum(hists.flatten())

    # Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists


#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    img_color_double_flat = np.reshape(img_color_double, (
        img_color_double.shape[0] * img_color_double.shape[1], img_color_double.shape[2]))
    bins = np.linspace(0, 255, num=num_bins + 1)

    # Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    # Loop for each pixel i in the image
    for i_pixel in range(img_color_double.shape[0] * img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        color = img_color_double_flat[i_pixel]
        coordinates = np.zeros(2, dtype=np.int)

        for i_color in range(color.shape[0] - 1):
            single_color = color[i_color]
            c = 0
            while c < num_bins - 1 and single_color >= bins[c + 1]:
                c += 1
            coordinates[i_color] = c
        hists[coordinates[0], coordinates[1]] += 1

    # Normalize the histogram such that its integral (sum) is equal 1
    hists = hists / sum(hists.flatten())

    # Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists


#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    sigma, caps = 3, (-6, 6)
    kernel = gauss_module.gaussdx(sigma)[0]
    kernel = kernel[int((len(kernel) - 1) / 2) + caps[0]: int((len(kernel) + 1) / 2) + caps[1]]
    kernel = (kernel / kernel.sum()).reshape(1, kernel.shape[0])

    imgDx = conv2(img_gray, kernel, 'same')
    imgDy = conv2(img_gray, kernel.transpose(), 'same')

    # rescaling to [0, 255]
    def scale(img):
        img_flat = img.flatten()
        img_scaled = ((img - min(img_flat)) / (max(img_flat) - min(img_flat))) * 255
        return img_scaled

    imgDx, imgDy, imgDz = scale(imgDx), scale(imgDy), np.zeros_like(imgDx)
    img_concat = np.zeros(shape=(imgDx.shape[0], imgDx.shape[1], 3))
    img_concat[:, :, 0], img_concat[:, :, 1], img_concat[:, :, 2] = imgDx, imgDy, imgDz

    # Define a 2D histogram  with "num_bins^2" number of entries
    hists = rg_hist(img_concat.astype('double'), num_bins)

    return hists


def is_grayvalue_hist(hist_name):
    if hist_name == 'grayvalue' or hist_name == 'dxdy':
        return True
    elif hist_name == 'rgb' or hist_name == 'rg':
        return False
    else:
        assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
    if hist_name == 'grayvalue':
        return normalized_hist(img, num_bins_gray)[0]
    elif hist_name == 'rgb':
        return rgb_hist(img, num_bins_gray)
    elif hist_name == 'rg':
        return rg_hist(img, num_bins_gray)
    elif hist_name == 'dxdy':
        return dxdy_hist(img, num_bins_gray)
    else:
        assert False, 'unknown distance: %s' % hist_name
