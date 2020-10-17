import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(query_images), len(model_images)))
    
    #... (your code here)
    for i in range(len(query_images)):
        for j in range(len(model_images)):
            D[i][j] = dist_module.get_dist_by_name(query_hists[i], model_hists[j], dist_type)
    
    best_match = D.argmax(axis=1) # or axis=0 ?!?!?!

    return best_match, D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []
    
    # Compute hisgoram for each image and add it at the bottom of image_hist
    #... (your code here)
    for image in image_list:
        if hist_isgray:
            img_color = np.array(Image.open(image))
            img = rgb2gray(img_color.astype('double'))
        else:
            img = np.array(Image.open(image)).astype('double')
        
        hist = histogram_module.get_hist_by_name(img, num_bins, hist_type)
        
        image_hist.append(hist)

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    plt.figure(1)

    num_nearest = 5  # show the top-5 neighbors

    #... (your code here)
    best_match, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    nearest_img_index = D.argsort(axis=1)[:,::-1][:,:num_nearest]
    
    for i in range(len(query_images)):
        # nearest for query_image i
        for j in range(num_nearest):
            idx_img = nearest_img_index[i][j]
            img_color = np.array(Image.open(model_images[idx_img]))
            plt.subplot(len(query_images), num_nearest, 1 + (i * num_nearest) + j)
            plt.imshow(img_color)
            
    plt.show()

