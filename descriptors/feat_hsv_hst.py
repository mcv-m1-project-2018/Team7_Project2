import cv2
import matplotlib.pyplot as plt
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


def get_hsv_hist(img,pyramid = [1,4,16,32,128,256], visualize=True):
    pyramid_slices =  get_im_pyramid(im_shape = img.shape, pyramid = pyramid)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pyramid_hsv = [ [] for i in range(len(pyramid)) ]
    #print(img.shape)
    #print(len(pyramid_slices), len(pyramid_slices[0]), pyramid_slices)

    for i, slices in enumerate(pyramid_slices):
        for slice in slices:
            x,y,w,h = slice
            crop = np.copy(hsv_image[x:x+w,y:y+h])
            #cv2.imshow('crop',crop)
            #cv2.waitKey()
            hsv_hist  = cv2.calcHist([crop], [0, 1, 2], None, [80, 5, 3], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hsv_hist,hsv_hist)
            hsv_hist  = hsv_hist.flatten()
            pyramid_hsv[i].append(hsv_hist)

    return pyramid_hsv


def compare_histograms(pyramid_hsv_1, pyramid_hsv_2, method):
    score = 0

    norm_term = [len(regions_size) for regions_size in pyramid_hsv_1]
    norm_term = sum(norm_term)
    if method == cv2.HISTCMP_INTERSECT or method == cv2.HISTCMP_CHISQR or method == cv2.HISTCMP_CORREL or \
            method == cv2.HISTCMP_BHATTACHARYYA or method == cv2.HISTCMP_KL_DIV or method == cv2.HISTCMP_INTERSECT:

        for regions_hists_1,regions_hists_2  in zip(pyramid_hsv_1,pyramid_hsv_2):
            sub_score = 0
            for hist1, hist2 in zip(regions_hists_1,regions_hists_2):
                sub_score += cv2.compareHist(hist1, hist2, method)
            score += sub_score#/len(regions_hists_1)

    return score/norm_term#/len(pyramid_hsv_1)


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
