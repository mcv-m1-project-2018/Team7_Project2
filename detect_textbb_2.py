import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_handler import Data
import ground_truth_text
from math import sqrt


def rescale_bbox_sat_val(bbox, im, mean_saturation, mean_value):
    x, y1, w, h = bbox
    x1 = x
    x2 = x1 + w
    y2 = y1 + h

    # recenter
    x1 = min(x1, im.shape[1] - x2)
    x2 = max(x2, im.shape[1] - x1)

    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hue_im, sat_im, val_im = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

    threshold_distance = 5
    starty = max(1, int(y1 - 0.8 * h))
    max_j = 0
    found = False
    while(not found):
        for j in range(int(0.8 * h )):
            dif = 0
            count = 0
            for i in range(w):
                dif += abs(sat_im[starty + j, x1 + i] - mean_saturation) + abs(val_im[starty + j, x1 + i] - mean_value)
                count+=1
            if ((not found) and  (dif/count < threshold_distance)):
                found = True
                max_j = j
        if (not found):
            threshold_distance += 5
    y1 = min(starty + max_j, y1)

    threshold_distance = 5
    endy = min(im.shape[0] - 1, int(y2 + 0.8 * h))
    max_j = 0
    found = False
    while(not found):
        for j in range(int(0.8 * h )):
            dif = 0
            count = 0
            for i in range(w):
                dif += abs(sat_im[endy - j, x1 + i] - mean_saturation) + abs(val_im[endy - j, x1 + i] - mean_value)
                count += 1
            if ((not found) and  dif/count < threshold_distance):
                found = True
                max_j = j
        if (not found):
            threshold_distance += 5
    y2 = max(endy - max_j, y2)

    threshold_distance = 5
    startx = max(1, int(x1 - 0.1 * w))
    max_i = 0
    found = False
    while(not found):
        for i in range(int(0.1 * w)):
            dif = 0
            count = 0
            for j in range(y2 - y1):
                dif += abs(sat_im[y1 + j, startx + i] - mean_saturation) + abs(val_im[y1 + j, startx + i] - mean_value)
                count += 1
            if ((not found) and  dif/count < threshold_distance):
                found = True
                max_i = i
        if(not found):
            threshold_distance += 5
    x1 = min(startx + max_i, x1)

    threshold_distance = 5
    endx = min(im.shape[1] - 1, int(x2 + 0.1 * w))
    max_i = 0
    found = False
    while(not found):
        for i in range(int(0.1 * w )):
            dif = 0
            count = 0
            for j in range(y2 - y1):
                dif += abs(sat_im[y1 + j, endx - i] - mean_saturation) + abs(val_im[y1 + j, endx - i] - mean_value)
                count += 1
                if ((not found) and dif / count < threshold_distance):
                    found = True
                    max_i = i
            if (not found):
                threshold_distance += 5
    x2 = max(endx - max_i, x2)

    return (x1, y1, x2, y2)


def rescale_bbox_gradient(bbox, im):
    x, y1, w, h = bbox
    x1 = x
    x2 = x1+w
    y2 = y1+h

    #recenter
    x1 = min(x1, im.shape[1]-x2)
    x2 = max(x2, im.shape[1]-x1)

    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hue_im, sat_im, val_im = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

    kernel_size = 5
    sat_im = cv2.GaussianBlur(sat_im, (kernel_size, kernel_size), 0)
    val_im = cv2.GaussianBlur(val_im, (kernel_size, kernel_size), 0)

    k_size = 3
    sat_sobelx = np.absolute(cv2.Sobel(sat_im, cv2.CV_64F, 1, 0, ksize=k_size))
    sat_sobely = np.absolute(cv2.Sobel(sat_im, cv2.CV_64F, 0, 1, ksize=k_size))

    val_sobelx = np.absolute(cv2.Sobel(val_im, cv2.CV_64F, 1, 0, ksize=k_size))
    val_sobely = np.absolute(cv2.Sobel(val_im, cv2.CV_64F, 0, 1, ksize=k_size))

    mix_sobelx = (sat_sobelx + val_sobelx)
    mix_sobely = (sat_sobely + val_sobely)

    starty = max(1, int(y1-0.8*h))
    max_g = 0
    max_j = 0
    for j in range (int(0.8*h - h*0.1)):
        #calculate gradient
        g = 0
        for i in range(w):
            g += mix_sobely[starty+j, x1+i]
        if(g > max_g):
            max_g = g
            max_j = j
    y1 = min(starty+max_j, y1)

    endy = min(im.shape[0]-1, int(y2 + 0.8 * h))
    max_g = 0
    max_j = 0
    for j in range(int(0.8*h - h*0.1)):
        # calculate gradient
        g = 0
        for i in range(w):
            g += mix_sobely[endy - j, x1 + i]
        if (g > max_g):
            max_g = g
            max_j = j
    y2 = max(endy - max_j, y2)

    startx = max(1, int(x1 - 0.1 * w))
    max_g = 0
    max_i = 0
    for i in range(int(0.1 * w - w * 0.01)):
        # calculate gradient
        g = 0
        for j in range(y2-y1):
            g += mix_sobelx[y1 + j, startx + i]
        if (g > max_g):
            max_g = g
            max_i = i
    x1 = min(startx + max_i, x1)

    endx = min(im.shape[1] - 1, int(x2 + 0.1 * w))
    max_g = 0
    max_i = 0
    for i in range(int(0.1 * w - w * 0.01)):
        # calculate gradient
        g = 0
        for j in range(y2-y1):
            g += mix_sobelx[y1+j, endx - i]
        if (g > max_g):
            max_g = g
            max_i = i
    x2 = max(endx - max_i, x2)





    """
    plt.subplot(231)
    plt.title("Image")
    plt.imshow(im)
    plt.subplot(232)
    plt.title("Saturation")
    plt.imshow(sat_im, cmap="gray")
    plt.subplot(233)
    plt.title("Value")
    plt.imshow(val_im, cmap="gray")
    plt.subplot(234)
    plt.title("Mix Gradient X")
    plt.imshow(mix_sobelx, cmap="gray")
    plt.subplot(235)
    plt.title("Mix Gradient Y")
    plt.imshow(mix_sobely, cmap="gray")
    """





    plt.show()




    return (x1, y1, x2, y2)



def check_bbox(bbox, is_from_black, im):
    x1, y1, w, h = bbox
    mask = np.zeros(im.shape[:2], np.uint8)
    cv2.rectangle(mask, (x1, y1), (x1+w, y1+h), (1), -1)

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    if(is_from_black):
        letters = black_filter(im)
    else:
        letters = white_filter(im)

    not_letters = cv2.bitwise_not(letters)
    mask = cv2.bitwise_and(not_letters, mask)

    value_sum = 0
    saturation_sum = 0
    count = 1
    for i in range(w):
        for j in range(h):
            if(mask[y1+j,x1+i]):
                value_sum += hsv[y1+j,x1+i,2]
                saturation_sum += hsv[y1+j,x1+i,1]
                count += 1

    mean_value = value_sum/count
    mean_saturation = saturation_sum/count

    print("mean_value", mean_value)
    print("mean_saturation", mean_saturation)
    print("is_from_black", is_from_black)

    if (is_from_black):
        check =  (mean_saturation < 180 and mean_value > 110)
    else:
        check =  (mean_saturation < 180 and mean_value < 165)

    """if (not check):

        plt.subplot(131)
        plt.title("mask")
        plt.imshow(mask, cmap='gray')
        plt.subplot(132)
        plt.title("im")
        plt.imshow(im)
        plt.subplot(133)
        plt.title("letters")
        plt.imshow(letters, cmap='gray')

        plt.show()"""

    return (check, mean_saturation, mean_value)


def intersection_over_union(bbox1, bbox2):
    # each bounding box should be given as (x1, y1, x2, y2). (x1, y1) the top left point, (x2, y2) the bottom right
    # intersection bbox
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area


def combine_boxes(box1, box2):
    # box = [x,y,w,h]
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = max(box1[0] + box1[2], box2[0] + box2[2]) - x
    h = max(box1[1] + box1[3], box2[1] + box2[3]) - y

    return [x, y, w, h]


def black_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sensitivity = 50
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([255, 255, sensitivity])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    # cv2.imshow('black',mask)

    return mask


def white_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sensitivity = 50
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # cv2.imshow('white',mask)

    return mask


def line_detection(im):
    lines_np = []
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kernel_size = 3
    gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    edges_im = cv2.Canny(gray, 10, 50)  # 50-100
    lines_im = edges_im.copy() * 0
    lines = cv2.HoughLinesP(edges_im, rho=1, theta=1 * np.pi / 180, threshold=20, \
                            minLineLength=50, maxLineGap=10)  # threshold = 100,minLength = 50,maxLineGap = 50

    if type(lines) == type(None):
        lines = []

    if len(lines):
        for i in range(lines.shape[0]):
            x1 = lines[i][0][0]
            y1 = lines[i][0][1]
            x2 = lines[i][0][2]
            y2 = lines[i][0][3]
            if abs(y1 - y2) < 5 or abs(x1 - x2) < 5:
                line_np = [np.array((x1, y1)), np.array((x2, y2))]
                lines_np.append(line_np)
                cv2.line(lines_im, (x1, y1), (x2, y2), 255, 4)

    # cv2.imshow('gray',gray)
    # cv2.imshow('edges_im',edges_im)
    # cv2.imshow('lines',lines_im)
    # print(len(lines))
    return lines_np, lines_im, edges_im


def detect_text(im_rgb, color_msk):
    text_box = [0, 0, 0, 0]
    score = -1000000
    candidates = []
    horizontal_th = 3

    test_im = color_msk.copy()
    test_im = cv2.cvtColor(test_im, cv2.COLOR_GRAY2BGR)

    im2, contours, hierarchy = cv2.findContours(color_msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and w < 300 and h > 10 and h < 100:
            candidates.append([x, y, w, h])
            cv2.rectangle(test_im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #print(y + h)

    if not len(candidates):
        return text_box, score
    candidates_score = [0] * len(candidates)
    for i, cand in enumerate(candidates):
        for cand_ in candidates:
            x, y, w, h = cand
            x_, y_, w_, h_ = cand_
            if (abs((y + h) - (y_ + h_)) < horizontal_th):
                candidates_score[i] += 1

    max_score = max(candidates_score)
    best_cand_idx = candidates_score.index(max_score)
    best_cand = candidates[best_cand_idx]
    x, y, w, h = best_cand
    best_union = best_cand

    for i, cand in enumerate(candidates):
        x_, y_, w_, h_ = cand
        if (abs((y + h) - (y_ + h_)) < horizontal_th):
            best_union = combine_boxes(best_union, cand)

    x, y, w, h = best_union

    # cv2.drawContours(test_im, contours, -1, (0,255,0), 3)
    # cv2.imshow('contours', test_im)

    center_score = -(abs(x + w / 2 - 500) / 500) * 10  # ==> -10 to 0

    score = max_score + center_score
    text_box = best_union

    print('[', center_score, score, ']')

    return text_box, score


def main():
    data = Data(database_dir='w5_BBDD_random', query_dir='w5_devel_random')
    gt = ground_truth_text.get_text_gt()
    iou_list = []
    # loop over database_imgs without overloading memory

    for im, im_name in data.database_imgs:
        x1gt, y1gt, x2gt, y2gt = gt[im_name]
        big_image = im.copy()
        h, w = im.shape[0:2]
        ratio = 1000 / w
        im = cv2.resize(im, (1000, int(ratio * h)))  # (ratio*w,ratio*h))
        # cv2.rectangle(im,(x1gt,y1gt),(x2gt,y2gt),(0,255,0),2)
        #line_detection(im)
        w_msk = white_filter(im)
        b_msk = black_filter(im)

        detected_bbox = False

        while (not detected_bbox):

            w_box, w_score = detect_text(im, w_msk)
            b_box, b_score = detect_text(im, b_msk)

            if w_score > b_score:
                text_box = w_box
                is_from_black = False
            else:
                text_box = b_box
                is_from_black = True


            x, y, w, h = text_box

            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 5)
            detected_bbox, mean_saturation, mean_value = check_bbox(text_box, is_from_black, im)
            if(not detected_bbox):
                if(is_from_black):
                    cv2.rectangle(b_msk, (x, y), (x + w, y + h), (0), -1)
                else:
                    cv2.rectangle(w_msk, (x, y), (x + w, y + h), (0), -1)
                #plt.imshow(im)
                #plt.show()

        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        rescaled_text_box = [int(i / ratio) for i in text_box]
        x, y, w, h = rescaled_text_box
        # ---------------------------------

        """y = int(y - 0.2 * h)
        h = int(h + 0.6 * h)

        # x = int(x - 0.2*h)
        # w = int(w + 0.4*h)
        x = int(x - 0.04 * w)
        w = int(w + 0.08 * w)"""
        text_box = [x, y, w, h]

        #rescaled_text_box = rescale_bbox_gradient(text_box, big_image)
        rescaled_text_box = rescale_bbox_sat_val(text_box, big_image, mean_saturation, mean_value)

        x1, y1, x2, y2 = rescaled_text_box

        cv2.rectangle(im, (int(x1*ratio), int(y1*ratio)), (int(x2*ratio), int(ratio*y2)), (0, 255, 0), 2)
        # ---------------------------------



        #print(rescaled_text_box)
        #print(gt[im_name])
        iou = intersection_over_union(rescaled_text_box, gt[im_name])

        print("iou: " + str(iou))
        iou_list.append(iou)

        if(iou<0.9):
            print("---------------------------------")
            print(im_name)
            plt.imshow(im)
            plt.show()
            print()

    print(iou_list)
    print(np.mean(iou_list))


if __name__ == "__main__":
    main()