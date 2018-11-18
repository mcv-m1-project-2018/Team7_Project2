import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_handler import Data
import ground_truth_text
from math import sqrt

def euclidean_dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def distance_rectangles(r1, r2):
    x1, y1, x2, y2 = r1
    X1, Y1, X2, Y2 = r2
    left = X2 < x1
    right = x2 < X1
    bottom = Y2 < y1
    top = y2 < Y1
    if top and left:
        return 10*euclidean_dist((x1, y2), (X2, Y1))
    elif left and bottom:
        return 10*euclidean_dist((x1, y1), (X2, Y2))
    elif bottom and right:
        return 10*euclidean_dist((x2, y1), (X1, Y2))
    elif right and top:
        return 10*euclidean_dist((x2, y2), (X1, Y1))
    elif left:
        return x1 - X2
    elif right:
        return X1 - x2
    elif bottom:
        return 10*(y1 - Y2)
    elif top:
        return 10*(Y1 - y2)
    else:             # rectangles intersect
        return 0.

def bboxes_from_mask(threshold_image):
    im2, contours, hierarchy = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cont in contours:
        rectangle = cv2.boundingRect(cont)
        x, y, w, h = rectangle
        if(h>0.01*threshold_image.shape[0] and h<0.15*threshold_image.shape[0]): #we are interested in letters mostly
            bboxes.append((x, y, x+w, y+h))
    return bboxes

def bboxes_black(hsv_image):
    threshblack = cv2.inRange(hsv_image, (0, 0, 0), (180, 50, 50))
    size = max(hsv_image.shape)
    kernel_size = int(size / 100)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(threshblack, cv2.MORPH_CLOSE, kernel)
    return bboxes_from_mask(closing)


def bboxes_white(hsv_image):
    threshwhite = cv2.inRange(hsv_image, (0, 0, 200), (180, 50, 255))
    size = max(hsv_image.shape)
    kernel_size = int(size / 100)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(threshwhite, cv2.MORPH_CLOSE, kernel)
    return bboxes_from_mask(closing)

def main():

    data = Data(database_dir= 'w5_BBDD_random', query_dir= 'w5_devel_random')
    gt = ground_truth_text.get_text_gt()

    # loop over database_imgs without overloading memory
    for im, im_name in data.database_imgs:

        hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        value_image = hsv_image[:,:,2]

        x1, y1, x2, y2 = gt[im_name]

        bboxesblack = bboxes_black(hsv_image)
        bboxeswhite = bboxes_white(hsv_image)

        image_white = cv2.inRange(hsv_image, (0, 0, 200), (180, 50, 255))
        image_black = cv2.inRange(hsv_image, (0, 0, 0), (180, 50, 50))

        size = max(hsv_image.shape)
        kernel_size = int(size / 100)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        image_black = cv2.morphologyEx(image_black, cv2.MORPH_CLOSE, kernel)
        image_white = cv2.morphologyEx(image_white, cv2.MORPH_CLOSE, kernel)

        for bbox in bboxesblack:
            cv2.rectangle(image_black, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (150), 10)

        for bbox in bboxeswhite:
            cv2.rectangle(image_white, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (150), 10)


        ################### Merge Bboxs (letters) #################
        """"
        parameter_letters = im.shape[1]/80

        finished = False
        while(not finished):
            print(len(bboxes))
            newboxes = []
            merged = set()
            for bbox1 in bboxes:
                if bbox1 not in merged:
                    for bbox2 in bboxes:
                        if bbox2 not in merged:
                            if ( not bbox1 == bbox2 ):
                                dist = distance_rectangles(bbox1, bbox2)

                                if(dist < parameter_letters ):

                                    height_merged = max(bbox1[3], bbox2[3])-min(bbox1[1], bbox2[1])
                                    if (height_merged <0.15*im.shape[0]):
                                        merged.add(bbox1)
                                        merged.add(bbox2)
                                        newboxes.append( (min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]), max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3]) ) )
                    if ( bbox1 not in merged):
                        newboxes.append(bbox1)
            bboxes = newboxes
            if(len(merged)== 0):
                finished = True
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(closing_bboxs, (x1, y1), (x2, y2), (150), 10)


        cv2.rectangle(threshblack, (x1, y1), (x2,y2), (150), 10)
        cv2.rectangle(threshwhite, (x1, y1), (x2, y2), (150), 10)

        x1, y1, x2, y2 = gt[im_name]
        integral_image_val = cv2.integral(value_image)
        x2 = min(x2, im.shape[1] - 2)
        y2 = min(y2, im.shape[0] - 2)
        sum_val = integral_image_val[y2 + 1, x2 + 1] + integral_image_val[y1, x1] - integral_image_val[y2 + 1, x1] - \
                  integral_image_val[y1, x2 + 1]

        area = (x2 - x1) * (y2 - y1)
        mean_val = sum_val / area
        if (mean_val < 120):  # dark background -> white letters
            target_integral_image = integral_threshwhite
            print("Dark background")
        if (mean_val > 120):  # bright background -> dark letters
            target_integral_image = integral_threshblack
            print("White background")
        """
        plt.subplot(131)
        plt.title("Image")
        plt.imshow(im)
        plt.subplot(132)
        plt.title("Black boxes")
        plt.imshow(image_black, cmap='gray')
        plt.subplot(133)
        plt.title("White Boxes")
        plt.imshow(image_white, cmap='gray')


        plt.show()


        #count_letters = target_integral_image[y2 + 1, x2 + 1] + target_integral_image[y1, x1] - \
        #                    target_integral_image[y2 + 1, x1] - target_integral_image[y1, x2 + 1]
        #filling_letter = count_letters / area
        #print("Filling letter:" + str(filling_letter))











if __name__ == "__main__":
    main()