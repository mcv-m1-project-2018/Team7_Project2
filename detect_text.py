import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_handler import Data
import ground_truth_text
import detect_letters

def intersection(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def intersection_over_union(bbox1, bbox2):
    #each bounding box should be given as (x1, y1, x2, y2). (x1, y1) the top left point, (x2, y2) the bottom right
    #intersection bbox
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])


    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    area1 = (bbox1[2]-bbox1[0] + 1) * (bbox1[3]-bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    union_area = area1 + area2 - intersection_area


    return intersection_area/union_area

def main():

    data = Data(database_dir= 'w5_BBDD_random', query_dir= 'w5_devel_random')

    gt = ground_truth_text.get_text_gt()

    #### parameters obtained analyzing ground truth ###################
    mean_aspect_ratio = 7.996836472972114
    std_aspect_ratio = 1.8974561127167842
    mean_length = 125.89268292682927
    std_length = 29.238100687737802
    mean_area = 2080.360975609756
    std_area = 794.6253398125554
    mean_filling_ratio = 0.13717846010306994
    std_fillin_ratio = 0.07545651641355072
    mean_saturation = 37.55115543136576
    std_saturation = 28.178800884826995
    mean_centered_distance = 0.0069118142084640425
    std_centered_distance = 0.002423878582904023

    iou_list = []
    # loop over database_imgs without overloading memory
    for im, im_name in data.database_imgs:

        ################ Resize ########################################################

        FinalSize = 200
        shape_max = max(im.shape)
        ratio = FinalSize / shape_max
        hsv_image_big = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        image_white = cv2.inRange(hsv_image_big, (0, 0, 200), (180, 50, 255)) / 255
        image_black = cv2.inRange(hsv_image_big, (0, 0, 0), (180, 50, 50)) / 255

        size = max(hsv_image_big.shape)
        kernel_size = int(size / 100)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        image_black = cv2.morphologyEx(image_black, cv2.MORPH_CLOSE, kernel)
        image_white = cv2.morphologyEx(image_white, cv2.MORPH_CLOSE, kernel)

        image_blackwhite = cv2.bitwise_or(image_black, image_white)

        integral_threshblack_big = cv2.integral(image_black)
        integral_threshwhite_big = cv2.integral(image_white)
        integral_blackwhite_big = cv2.integral(image_blackwhite)

        im = cv2.resize(im, (0, 0), fx=ratio, fy=ratio)

        hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        rgb_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        hue_im, sat_im, val_im = hsv_image[:,:,0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        bboxes_white = detect_letters.bboxes_white(hsv_image_big)
        bboxes_black = detect_letters.bboxes_black(hsv_image_big)

        ################ Smoothing #####################################################

        kernel_size = 3


        #sat_im = cv2.GaussianBlur(sat_im, (kernel_size, kernel_size), 0)
        #val_im = cv2.GaussianBlur(val_im, (kernel_size, kernel_size), 0)


        ################ Gradient ######################################################

        k_size = 3

        hue_sobelx = np.absolute(cv2.Sobel(hue_im, cv2.CV_64F, 1, 0, ksize=k_size))
        hue_sobely = np.absolute(cv2.Sobel(hue_im, cv2.CV_64F, 0, 1, ksize=k_size))
        hue_sobel = hue_sobelx + hue_sobely

        sat_sobelx = np.absolute(cv2.Sobel(sat_im, cv2.CV_64F, 1, 0, ksize=k_size))
        sat_sobely = np.absolute(cv2.Sobel(sat_im, cv2.CV_64F, 0, 1, ksize=k_size))
        sat_sobel = sat_sobelx + sat_sobely

        val_sobelx = np.absolute(cv2.Sobel(val_im, cv2.CV_64F, 1, 0, ksize=k_size))
        val_sobely = np.absolute(cv2.Sobel(val_im, cv2.CV_64F, 0, 1, ksize=k_size))
        val_sobel = val_sobelx + val_sobely

        mix_sobelx = (sat_sobelx + val_sobelx) / 510
        mix_sobely = (sat_sobely + val_sobely) / 510
        mix_sobel = sat_sobel + val_sobel

        integral_mix_sobelx = cv2.integral(mix_sobelx)
        integral_mix_sobely = cv2.integral(mix_sobely)

        mix_sobel = mix_sobel#*255/np.max(mix_sobel)

        mix_sobel = mix_sobel.astype(np.uint8)

        ############## Filter High Saturation ################

        mask = cv2.inRange(sat_im, 0, 70)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)/255
        mix_sobelx = np.multiply(mix_sobelx, mask)
        mix_sobely = np.multiply(mix_sobely, mask)



        ############# Edges Custom #############################
        edges_step = 2
        soft_threshold_multiplier = 2
        high_threshold_multiplier = 4

        #horizontal edges
        hedges = []
        (mu, sigma) = cv2.meanStdDev(mix_sobely)
        soft_threshold = mu + soft_threshold_multiplier*sigma
        strong_threshold = mu + high_threshold_multiplier*sigma
        edge_state = -1  # -1 No edge, 0 soft edge, 1 strong edge
        for y in range(1, im.shape[0]-1, edges_step):
            for x in range(0, im.shape[1], edges_step):
                score = mix_sobely[y,x]+mix_sobely[y-1,x]+mix_sobely[y+1,x]
                if(edge_state == -1):
                    if(score>soft_threshold):
                        edge_state=0
                        edge_start = (x, y)

                if (edge_state == 0):
                    if(score>strong_threshold):
                        edge_state = 1
                    if (score < soft_threshold):
                        edge_state = -1

                if (edge_state == 1):
                    if (score < soft_threshold):
                        hedges.append((edge_start, (x, y)))
                        edge_state = -1

            if (edge_state == 1):
                hedges.append((edge_start, (x, y)))
            edge_state = -1

        for edge in hedges:
            cv2.line(img=rgb_image, pt1=edge[0], pt2=edge[1], color=(0,255,0), thickness=1)

        #print(len(hedges))

        # vertical edges
        vedges = []
        (mu, sigma) = cv2.meanStdDev(mix_sobelx)
        soft_threshold = mu + soft_threshold_multiplier * sigma
        strong_threshold = mu + high_threshold_multiplier * sigma
        edge_state = -1  # -1 No edge, 0 soft edge, 1 strong edge
        for x in range(1, im.shape[1]-1, edges_step):
            for y in range(0, im.shape[0], edges_step):
                score = mix_sobelx[y, x] + mix_sobelx[y, x-1] + mix_sobelx[y, x+1]
                if (edge_state == -1):
                    if (score > soft_threshold):
                        edge_state = 0
                        edge_start = (x, y)

                if (edge_state == 0):
                    if (score > strong_threshold):
                        edge_state = 1
                    if (score < soft_threshold):
                        edge_state = -1

                if (edge_state == 1):
                    if (score < soft_threshold):
                        vedges.append((edge_start, (x, y)))
                        edge_state = -1

            if (edge_state == 1):
                vedges.append((edge_start, (x, y)))
            edge_state = -1

        for edge in vedges:
            cv2.line(img=rgb_image, pt1=edge[0], pt2=edge[1], color=(255, 0, 0), thickness=1)

        #print(len(vedges))
        ############# "Candidate Windows from edge vertices ####################
        candidate_points = []
        for edge in hedges:
            candidate_points.append(edge[0])
            candidate_points.append(edge[1])
        for edge in vedges:
            candidate_points.append(edge[0])
            candidate_points.append(edge[1])
        #print("len candidate points")
        #print(len(candidate_points))

        integral_image_sat = cv2.integral(sat_im)
        integral_image_val = cv2.integral(val_im)

        ################ GT test ################################################

        x1_big, y1_big, x2_big, y2_big = gt[im_name]

        point1 = (int(x1_big * ratio), int(y1_big * ratio))
        point2 = (int(x2_big * ratio), int(y2_big * ratio))

        passed = False

        length = abs(point1[0] - point2[0])
        if (length > 40 and length < 170):

            height = abs(point1[1] - point2[1])
            if (height > 0):

                aspect_ratio = length / height
                if (aspect_ratio > 3 and aspect_ratio < 14):

                    area = length * height
                    if (area > 400 and area < 5000):

                        x1 = min(point1[0], point2[0])
                        y1 = min(point1[1], point2[1])
                        x2 = max(point1[0], point2[0])
                        y2 = max(point1[1], point2[1])

                        x2 = min(x2, im.shape[1] - 1)
                        y2 = min(y2, im.shape[0] - 1)

                        sum_sat = integral_image_sat[y2 + 1, x2 + 1] + integral_image_sat[y1, x1] - \
                                  integral_image_sat[y2 + 1, x1] - integral_image_sat[y1, x2 + 1]
                        mean_sat = sum_sat / area

                        if (mean_sat < 150):

                            x1_big, y1_big, x2_big, y2_big = int(x1 / ratio), int(y1 / ratio), int(
                                x2 / ratio), int(y2 / ratio)

                            centered_distance = abs(x1_big - (hsv_image_big.shape[1] - x2_big)) / hsv_image_big.shape[1]

                            if (centered_distance < 0.2):

                                sum_val = integral_image_val[y2 + 1, x2 + 1] + integral_image_val[y1, x1] - \
                                          integral_image_val[y2 + 1, x1] - integral_image_val[y1, x2 + 1]
                                mean_val = sum_val / area

                                if (mean_val < 120):  # dark background -> white letters
                                    target_integral_image = integral_threshwhite_big
                                    is_dark_bg = True
                                else:
                                    is_dark_bg = False
                                if (mean_val > 140):  # bright background -> dark letters
                                    target_integral_image = integral_threshblack_big
                                    is_bright_bg = True
                                else:
                                    is_bright_bg = False
                                if (not (is_dark_bg or is_bright_bg)):
                                    target_integral_image = integral_blackwhite_big


                                count_letters_big = target_integral_image[y2_big + 1, x2_big + 1] + \
                                                    target_integral_image[y1_big, x1_big] - \
                                                    target_integral_image[y2_big + 1, x1_big] - \
                                                    target_integral_image[y1_big, x2_big + 1]
                                area_big = area / ratio / ratio
                                filling_letters = count_letters_big / area_big

                                if (filling_letters > 0.02 and filling_letters < 0.4):
                                    passed = True

                                    #################### SCORE WINDOWS AND RETRIEVE BEST ##########################

                                    distance = abs(aspect_ratio - mean_aspect_ratio) / std_aspect_ratio + \
                                               abs(length - mean_length) / std_length + \
                                               abs(area - mean_area) / std_area + \
                                               abs(filling_letters - mean_filling_ratio) / std_fillin_ratio + \
                                               abs(mean_sat - mean_saturation) / std_saturation

                                    x2 = min(x2, im.shape[1] - 2)
                                    y2 = min(y2, im.shape[0] - 2)
                                    x1 = max(1, x1)
                                    y1 = max(1, y1)

                                    count_gradient_htop = (integral_mix_sobely[y1 + 2, x2 + 1] + \
                                                           integral_mix_sobely[y1 - 1, x1] - \
                                                           integral_mix_sobely[y1 - 1, x2 + 1] - \
                                                           integral_mix_sobely[y1 + 2, x1]) / ((x2 - x1) * 3)

                                    count_gradient_hbot = (integral_mix_sobely[y2 + 2, x2 + 1] + \
                                                           integral_mix_sobely[y2 - 1, x1] - \
                                                           integral_mix_sobely[y2 - 1, x2 + 1] - \
                                                           integral_mix_sobely[y2 + 2, x1]) / ((x2 - x1) * 3)

                                    count_gradient_vleft = (integral_mix_sobelx[y2 + 1, x1 + 2] + \
                                                            integral_mix_sobelx[y1, x1 - 1] - \
                                                            integral_mix_sobelx[y1, x1 + 2] - \
                                                            integral_mix_sobelx[y2 + 1, x1 - 1]) / ((y2 - y1) * 3)

                                    count_gradient_right = (integral_mix_sobelx[y2 + 1, x2 + 2] + \
                                                            integral_mix_sobelx[y1, x2 - 1] - \
                                                            integral_mix_sobelx[y1, x2 + 2] - \
                                                            integral_mix_sobelx[y2 + 1, x2 - 1]) / ((y2 - y1) * 3)

                                    gradient_score = count_gradient_htop + count_gradient_hbot + count_gradient_vleft + count_gradient_right

                                    if (is_dark_bg):
                                        bboxes = bboxes_white
                                    elif (is_bright_bg):
                                        bboxes = bboxes_black
                                    else:
                                        bboxes = bboxes_white + bboxes_black

                                    box_cuts = 0
                                    intersections = 0
                                    for bbox in bboxes:
                                        intersect = intersection(bbox, (x1_big, y1_big, x2_big, y2_big))
                                        area_bbox = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                                        cut = min(intersect, area_bbox - intersect)
                                        box_cuts += cut
                                        intersections += intersect

                                    intersect_score = intersections / area_big
                                    box_cuts = box_cuts / area_big

                                    gt_dict_score = {}
                                    gt_dict_score["gradient_score"] = gradient_score
                                    gt_dict_score["box_cuts"] = box_cuts
                                    gt_dict_score["dist_aspect_ratio"] = abs(aspect_ratio - mean_aspect_ratio) / std_aspect_ratio
                                    gt_dict_score["dist_length"] = abs(length - mean_length) / std_length
                                    gt_dict_score["area"] = abs(area - mean_area) / std_area
                                    gt_dict_score["filling_letters"] = abs(filling_letters - mean_filling_ratio) / std_fillin_ratio
                                    gt_dict_score["saturation"] = abs(mean_sat - mean_saturation) / std_saturation
                                    gt_dict_score["intersect score"] = intersect_score

                                    gt_score = distance/4 - gradient_score + 10*box_cuts - 5*intersect_score

        if(passed and (intersect_score>0)):
            print("Test gt ok: " + im_name)
        else:
            print ("---------------")
            print("TEST GT not PASSED!!!: "+im_name)
            print("intersect_score", intersect_score)
            print("length", length)
            print("aspect_ratio", aspect_ratio)
            print("area", area)
            print("mean_sat", mean_sat)
            print("centered", centered_distance)
            print("filling ratio", filling_letters)
            print("is_dark", is_dark_bg)
            print("is_bright", is_bright_bg)
            print(mean_val)
            print("... Bboxes")
            print((x1_big, y1_big, x2_big, y2_big))
            print(bboxes)
            for bbox in bboxes:
                cv2.rectangle(image_black, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (1), 10)
                cv2.rectangle(image_white, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (1), 10)
            plt.subplot(331)
            plt.title("Image")
            plt.imshow(rgb_image)
            plt.subplot(332)
            plt.title("Saturation")
            plt.imshow(sat_im, cmap='gray')
            plt.subplot(333)
            plt.title("Value")
            plt.imshow(val_im, cmap='gray')

            plt.subplot(3, 3, 7)
            plt.title("Threshold_Black")
            plt.imshow(image_black, cmap='gray')

            plt.subplot(3, 3, 8)
            plt.title("Threshold_White")
            plt.imshow(image_white, cmap='gray')

            plt.subplot(3, 3, 5)
            plt.title("Saturation Gradient")
            plt.imshow(sat_sobel, cmap='gray')

            plt.subplot(3, 3, 6)
            plt.title("Value Gradient")
            plt.imshow(val_sobel, cmap='gray')

            plt.subplot(3, 3, 4)
            plt.title("Mask")
            plt.imshow(mask, cmap='gray')

            plt.show()

        cv2.rectangle(rgb_image, pt1=point1, pt2=point2, color=(0, 255, 0), thickness=2)

        count = 0
        window_candidates = []
        window_candidates
        for point1 in candidate_points:
            for point2 in candidate_points:

                ######################################## FILTER WINDOWS ###############################################
                length = abs(point1[0]-point2[0])
                if(length >40 and length < 170):

                    height = abs(point1[1] - point2[1])
                    if(height > 0):

                        aspect_ratio = length / height
                        if (aspect_ratio>3 and aspect_ratio<14):

                            area = length*height
                            if(area > 400 and area < 5000):

                                x1 = min(point1[0], point2[0])
                                y1 = min(point1[1], point2[1])
                                x2 = max(point1[0], point2[0])
                                y2 = max(point1[1], point2[1])

                                x2 = min(x2, im.shape[1] - 1)
                                y2 = min(y2, im.shape[0] - 1)

                                x1_big, y1_big, x2_big, y2_big = int(x1 / ratio), int(y1 / ratio), int(
                                    x2 / ratio), int(y2 / ratio)

                                sum_sat = integral_image_sat[y2 + 1, x2 + 1] + integral_image_sat[y1, x1] - \
                                          integral_image_sat[y2 + 1, x1] - integral_image_sat[y1, x2 + 1]
                                mean_sat = sum_sat / area

                                if(mean_sat < 150):

                                    centered_distance = abs(x1_big - (hsv_image_big.shape[1] - x2_big)) / hsv_image_big.shape[1]

                                    if(centered_distance < 0.2):

                                        sum_val = integral_image_val[y2 + 1, x2 + 1] + integral_image_val[y1, x1] - \
                                                  integral_image_val[y2 + 1, x1] - integral_image_val[y1, x2 + 1]
                                        mean_val = sum_val / area

                                        if (mean_val < 120):  # dark background -> white letters
                                            target_integral_image = integral_threshwhite_big
                                            is_dark_bg = True
                                        else:
                                            is_dark_bg = False
                                        if (mean_val > 140):  # bright background -> dark letters
                                            target_integral_image = integral_threshblack_big
                                            is_bright_bg = True
                                        else:
                                            is_bright_bg = False
                                        if (not (is_dark_bg or is_bright_bg)):
                                            target_integral_image = integral_blackwhite_big


                                        count_letters_big = target_integral_image[y2_big + 1, x2_big + 1] + \
                                                            target_integral_image[y1_big, x1_big] - \
                                                            target_integral_image[y2_big + 1, x1_big] - \
                                                            target_integral_image[y1_big, x2_big + 1]
                                        area_big = area / ratio / ratio
                                        filling_letters = count_letters_big / area_big

                                        if (filling_letters > 0.02 and filling_letters < 0.4):


                                            count+=1
                                            #cv2.rectangle(rgb_image, pt1=point1, pt2=point2, color=(0,0,255), thickness=1)

                                            #################### SCORE WINDOWS AND RETRIEVE BEST ##########################

                                            distance = abs(aspect_ratio-mean_aspect_ratio)/std_aspect_ratio + \
                                                abs(length-mean_length)/std_length + \
                                                abs(area-mean_area)/std_area + \
                                                abs(filling_letters-mean_filling_ratio)/std_fillin_ratio + \
                                                abs(mean_sat-mean_saturation)/std_saturation

                                            x2 = min(x2, im.shape[1] - 2)
                                            y2 = min(y2, im.shape[0] - 2)
                                            x1 = max(1, x1)
                                            y1 = max(1, y1)

                                            count_gradient_htop = (integral_mix_sobely[y1 + 2, x2 + 1] + \
                                                                  integral_mix_sobely[y1-1, x1] - \
                                                                  integral_mix_sobely[y1-1 , x2+1] - \
                                                                  integral_mix_sobely[y1+2, x1]) / ((x2-x1)*3)

                                            count_gradient_hbot = (integral_mix_sobely[y2 + 2, x2 + 1] + \
                                                                  integral_mix_sobely[y2 - 1, x1] - \
                                                                  integral_mix_sobely[y2 - 1, x2 + 1] - \
                                                                  integral_mix_sobely[y2 + 2, x1]) / ((x2-x1)*3)

                                            count_gradient_vleft = (integral_mix_sobelx[y2 + 1, x1 + 2] + \
                                                                  integral_mix_sobelx[y1, x1-1] - \
                                                                  integral_mix_sobelx[y1, x1 + 2] - \
                                                                  integral_mix_sobelx[y2 + 1, x1-1]) / ((y2-y1)*3)

                                            count_gradient_right = (integral_mix_sobelx[y2 + 1, x2 + 2] + \
                                                                   integral_mix_sobelx[y1, x2 - 1] - \
                                                                   integral_mix_sobelx[y1, x2 + 2] - \
                                                                   integral_mix_sobelx[y2 + 1, x2 - 1]) / ((y2-y1)*3)

                                            gradient_score = count_gradient_htop + count_gradient_hbot + count_gradient_vleft + count_gradient_right

                                            if(is_dark_bg):
                                                bboxes = bboxes_white
                                            elif(is_bright_bg):
                                                bboxes = bboxes_black
                                            else: bboxes = bboxes_white+bboxes_black

                                            box_cuts = 0
                                            intersections = 0
                                            for bbox in bboxes:
                                                intersect = intersection(bbox, (x1_big, y1_big, x2_big, y2_big))
                                                area_bbox = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                                                cut = min(intersect, area_bbox-intersect)
                                                box_cuts += cut
                                                intersections += intersect

                                            intersect_score = intersections / area_big
                                            box_cuts = box_cuts / area_big

                                            dict_score = {}
                                            dict_score["gradient_score"] = gradient_score
                                            dict_score["box_cuts"] = box_cuts
                                            dict_score["dist_aspect_ratio"] = abs(aspect_ratio-mean_aspect_ratio)/std_aspect_ratio
                                            dict_score["dist_length"] = abs(length-mean_length)/std_length/4
                                            dict_score["area"] = abs(area-mean_area)/std_area/4
                                            dict_score["filling_letters"] = abs(filling_letters-mean_filling_ratio)/std_fillin_ratio
                                            dict_score["saturation"] = abs(mean_sat - mean_saturation) / std_saturation
                                            dict_score["intersect score"] = intersect_score

                                            score = distance/4 - gradient_score + 10*box_cuts - 5*intersect_score

                                            window_candidates.append(((x1, y1, x2, y2), score, dict_score))


        #print("candidate windows")
        #print(count)
        if(len(window_candidates)==0):
            print("Empty windows candidates!!!: "+im_name)
            winning_window = (0,0,0,0)
        else:
            winning_window, scoreBest, dict_scoreBest = min(window_candidates, key=lambda x: x[1])
        #print("WINNER")
        #print(winning_window)

        gt_window = gt[im_name]






        #print("winning window", winning_window)
        winning_window_big = ( int(winning_window[0]/ratio), int(winning_window[1]/ratio), int(winning_window[2]/ratio), int(winning_window[3]/ratio) )

        #print("winning window", winning_window_big)
        #print("gt window", gt_window)

        iou = intersection_over_union(winning_window_big, gt_window)
        print("iou: "+str(iou))
        iou_list.append(iou)


        point1 = (int(gt_window[0] * ratio), int(gt_window[1] * ratio))
        point2 = (int(gt_window[2] * ratio), int(gt_window[3] * ratio))

        p1 = winning_window[:2]
        p2 = winning_window[2:]
        #print(p1)
        #print(p2)
        cv2.rectangle(rgb_image, pt1=p1, pt2=p2, color=(0, 0, 255), thickness=2)
        #print("-----------------------------")
        if(gt_score < scoreBest):
            print("GT WINS")
        else:
            print("GT Loses")
            print("-----------------------------------------")
            print("score winning window", scoreBest)
            print("scores winning window", dict_scoreBest)
            print("...................")
            print("gt score", gt_score)
            print("gt scores", gt_dict_score)
        if(iou < 0.4):
            plt.subplot(331)
            plt.title("Image")
            plt.imshow(rgb_image)
            plt.subplot(332)
            plt.title("Saturation")
            plt.imshow(sat_im, cmap='gray')
            plt.subplot(333)
            plt.title("Value")
            plt.imshow(val_im, cmap='gray')

            plt.subplot(3, 3, 7)
            plt.title("Threshold_Black")
            plt.imshow(image_black, cmap='gray')

            plt.subplot(3, 3, 8)
            plt.title("Threshold_White")
            plt.imshow(image_white, cmap='gray')

            plt.subplot(3, 3, 5)
            plt.title("Saturation Gradient")
            plt.imshow(sat_sobel, cmap='gray')

            plt.subplot(3, 3, 6)
            plt.title("Value Gradient")
            plt.imshow(val_sobel, cmap='gray')

            plt.subplot(3, 3, 4)
            plt.title("Mask")
            plt.imshow(mask, cmap='gray')



            plt.show()
        """"

        ############## Edges Canny ##############################


        (mu, sigma) = cv2.meanStdDev(mix_sobel)
        edges = cv2.Canny(mix_sobel, mu - 2*sigma, mu + sigma)
        edges = np.multiply(edges, mask)

        ############### Corners ###############################

        sat_corners = cv2.cornerHarris(sat_im, blockSize=2, ksize=5, k=0.01)
        val_corners = cv2.cornerHarris(val_im, blockSize=2, ksize=5, k=0.01)

        # Threshold for an optimal value, it may vary depending on the image.
        #rgb_image[dst > 0.2 * dst.max()] = [0, 0, 255]

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)

        for line in lines:
            print (line)
            print("-------")

        for line in lines:
            rho, theta = line[0]
            if(theta==0):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 500 * (-b))
                y1 = int(y0 + 500 * (a))
                x2 = int(x0 - 500 * (-b))
                y2 = int(y0 - 500 * (a))

                cv2.line(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        ret, thresh = cv2.threshold(edges, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            area1 = w*h
            area2 = cv2.contourArea(cont)
            if(area2 > 0):
                if (float(area1)/float(area2) <1.2 and float(area1)/float(area2) > 0.8


                    ):
                    cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


        ############### Sliding window ##################################################
        #Values obtained analizing ground truth

        minLength = FinalSize/2
        maxLength = FinalSize*1.2
        stepLength = FinalSize/20

        minAspectRatio = 4
        maxAspectRatio = 13
        stepAspectRatio = 0.5

        for len in range(minLength, maxLength, stepLength):
            for ar in range(minAspectRatio, maxAspectRatio, stepAspectRatio):
                for i in range(1, im.shape[0]*ratio, 2):
                    for j in range(1, im.shape[1] * ratio, 2):








        plt.subplot(331)
        plt.title("Image")
        plt.imshow(rgb_image)
        plt.subplot(332)
        plt.title("Saturation")
        plt.imshow(sat_im, cmap='gray')
        plt.subplot(333)
        plt.title("Value")
        plt.imshow(val_im, cmap='gray')

        plt.subplot(3, 3, 7)
        plt.title("Mix Gradient X")
        plt.imshow(mix_sobelx, cmap='gray')

        plt.subplot(3, 3, 8)
        plt.title("Mix Gradient Y")
        plt.imshow(mix_sobely, cmap='gray')

        plt.subplot(3, 3, 5)
        plt.title("Saturation Gradient")
        plt.imshow(sat_sobel, cmap='gray')

        plt.subplot(3, 3, 6)
        plt.title("Value Gradient")
        plt.imshow(val_sobel, cmap='gray')

        plt.subplot(3, 3, 4)
        plt.title("Mask")
        plt.imshow(mask, cmap='gray')

        plt.subplot(3, 4, 8)
        plt.title("Edges")
        plt.imshow(edges, cmap='gray')

        plt.subplot(3, 4, 9)
        plt.title("Corners Sat")
        plt.imshow(sat_corners, cmap='gray')

        plt.subplot(3, 4, 10)
        plt.title("Corners Val")
        plt.imshow(val_corners, cmap='gray')


        plt.show()
        """

    print("------------------------------------------")
    print("MEAN INTERSECTION OVER UNION")
    print(iou_list)
    print (np.mean(iou_list))




if __name__ == "__main__":
    main()