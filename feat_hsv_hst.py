import cv2
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import numpy as np


def get_nonoverllaped_regions(im_shape, num_regions):
    im_width,im_height,_ = im_shape
    slices       = []
    window_width  = int( im_width / np.sqrt(num_regions) )
    window_height = int( im_height/ np.sqrt(num_regions) )    

    for x in range(0, im_width-window_width+1, window_width):
        for y in range(0, im_height-window_height+1, window_height):
            slices.append([x,y,window_width,window_height])
    return slices

def get_im_pyramid(im_shape, pyramid):
    pyramid_slices       = []
    
    for num_regions in pyramid:
        pyramid_slices.append( get_nonoverllaped_regions(im_shape, num_regions) )

    return pyramid_slices

"""
def get_hsv(img, visualize=False):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    histH = cv2.calcHist(images=[hsv_image], channels=[0], mask=None, histSize=[150], ranges=[0, 180])
    histS = cv2.calcHist([hsv_image], [1], None, [20], [0, 256])
    histV = cv2.calcHist([hsv_image], [2], None, [20], [0, 256])
    sum_hist = histH.sum()

    feats = {'histH': histH / sum_hist,
            'histS': histS  / sum_hist,
            'histV': histV  / sum_hist}

    if (visualize):
        RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(221), plt.imshow(RGB_image)
        plt.subplot(222), plt.plot(histH / sum_hist, )
        plt.subplot(223), plt.plot(histS / sum_hist)
        plt.subplot(224), plt.plot(histV / sum_hist)
        plt.show()

    return feats

def get_hsv_hist(img, visualize=True):    
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist  = cv2.calcHist([hsv_image], [0, 1, 2], None, [100, 10, 10], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hsv_hist,hsv_hist)
    hsv_hist  = hsv_hist.flatten()

    return hsv_hist
"""

def get_hsv_hist(img,pyramid = [200], visualize=True):
    pyramid_slices =  get_im_pyramid(im_shape = img.shape, pyramid = pyramid)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pyramid_hsv = [ [] for i in range(len(pyramid)) ]
    #print(img.shape)
    #print(len(pyramid_slices), len(pyramid_slices[0]), pyramid_slices)

    for i, slices in enumerate(pyramid_slices):
        for slice in slices:
            x,y,w,h = slice
            crop = hsv_image[x:x+w,y:y+h]
            #cv2.imshow('crop',crop)
            #cv2.waitKey()
            hsv_hist  = cv2.calcHist([crop], [0, 1, 2], None, [80, 5, 3], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hsv_hist,hsv_hist)
            hsv_hist  = hsv_hist.flatten()
            pyramid_hsv[i].append(hsv_hist)

    return pyramid_hsv


def compare_histograms(pyramid_hsv_1, pyramid_hsv_2, method):
    score = 0
    if method == cv2.HISTCMP_INTERSECT or method == cv2.HISTCMP_CHISQR or method == cv2.HISTCMP_CORREL or \
            method == cv2.HISTCMP_BHATTACHARYYA or method == cv2.HISTCMP_KL_DIV or method == cv2.HISTCMP_INTERSECT:

        for regions_hists_1,regions_hists_2  in zip(pyramid_hsv_1,pyramid_hsv_2):
            sub_score = 0
            for hist1, hist2 in zip(regions_hists_1,regions_hists_2):
                sub_score += cv2.compareHist(hist1, hist2, method)
            score += sub_score/len(regions_hists_1)

    if method == 'emd':
        score = wasserstein_distance(hist1[:, 0], hist2[:, 0])  # earth mover's distance

    return score


def retrieve_best_results(image_histH, database_imgs, database_hist, method=cv2.HISTCMP_BHATTACHARYYA, K=10):
    """
    Call this function in order to compare a histogram with the rest of the dataset
    :param image_histH: hsv histogram of the image to match
    :param database_imgs: list of [numpy_rgb_image, image_name]
    :param method:
    :param K:
    :return: the k top matches
    """
    scores = []
    for [d_im, d_name], d_hist in zip(database_imgs, database_hist):
        #scores.append( ( d_name , difference(image_histH, d_hist['histH'])) )
        scores.append((d_name, compare_histograms(image_histH, d_hist, method=method)))

    if method in [cv2.HISTCMP_INTERSECT, cv2.HISTCMP_CORREL]:
        scores.sort(key=lambda s: s[1], reverse=True)
    else:
        scores.sort(key=lambda s: s[1], reverse=False)

    return scores[:K]
