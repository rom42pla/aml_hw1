import numpy as np
import math



# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):
    minimum = np.minimum(x,y).sum()
    d = ((minimum / x.sum()) + (minimum / y.sum())) / 2
    
    if (d >= 0) and (d <=1):
        return (1 - d)
    else:
        raise ValueError('the distance is not between 0 and 1')

# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):
    d = np.power(x-y,2).sum()
    
    if (d >= 0) and (d <=np.sqrt(2)):
        return d
    else:
        raise ValueError('the distance is not between 0 and sqrt(2)')



# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):
    x = x + 1
    y = y + 1
    
    d = (np.power(x-y,2) / (x+y)).sum()
    
    if (d >= 0):
        return d
    else:
        raise ValueError('the distance is not between 0 and Inf')



def get_dist_by_name(x, y, dist_name):
    if dist_name == 'chi2':
        return dist_chi2(x,y)
    elif dist_name == 'intersect':
        return dist_intersect(x,y)
    elif dist_name == 'l2':
        return dist_l2(x,y)
    else:
        assert False, 'unknown distance: %s'%dist_name
  




