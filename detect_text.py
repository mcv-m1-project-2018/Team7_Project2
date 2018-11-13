import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_handler import Data
import ground_truth_text

def main():

    data = Data(database_dir= 'w5_BBDD_random', query_dir= 'w5_devel_random')

    gt = ground_truth_text.get_text_gt()

    # loop over database_imgs without overloading memory
    for im, im_name in data.database_imgs:

        ################ Resize ########################################################

        FinalSize = 200
        shape_max = max(im.shape)
        ratio = FinalSize / shape_max
        hsv_image_big = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        threshblack_big = cv2.inRange(hsv_image_big, (0, 0, 0), (180, 50, 50))
        threshwhite_big = cv2.inRange(hsv_image_big, (0, 0, 200), (180, 50, 255))

        integral_threshblack_big = cv2.integral(threshblack_big)/255
        integral_threshwhite_big = cv2.integral(threshwhite_big)/255

        im = cv2.resize(im, (0, 0), fx=ratio, fy=ratio)

        hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        rgb_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        hue_im, sat_im, val_im = hsv_image[:,:,0], hsv_image[:, :, 1], hsv_image[:, :, 2]

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

        #for edge in hedges:
            #cv2.line(img=rgb_image, pt1=edge[0], pt2=edge[1], color=(0,255,0), thickness=1)

        print(len(hedges))

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

        #for edge in vedges:
            #cv2.line(img=rgb_image, pt1=edge[0], pt2=edge[1], color=(255, 0, 0), thickness=1)

        print(len(vedges))
        ############# "Candidate Windows from edge vertices ####################
        candidate_points = []
        for edge in hedges:
            candidate_points.append(edge[0])
            candidate_points.append(edge[1])
        for edge in vedges:
            candidate_points.append(edge[0])
            candidate_points.append(edge[1])
        print("len candidate points")
        print(len(candidate_points))

        integral_image_sat = cv2.integral(sat_im)
        integral_image_val = cv2.integral(val_im)

        ################ GT test ################################################

        x1gt, y1gt, x2gt, y2gt = gt[im_name]

        point1 = (int(x1gt*ratio), int(y1gt*ratio))
        point2 = (int(x2gt*ratio), int(y2gt*ratio))
        print("--------------------------------------------")
        length = abs(point1[0] - point2[0])
        print("length " + str(length))
        if (length > 50 and length < 160):
            height = abs(point1[1] - point2[1])
            print("height "+str(height))
            if (height > 0):
                aspect_ratio = length / abs(point1[1] - point2[1])
                print("aspect_ratio"+str(aspect_ratio))
                if (aspect_ratio > 4 and aspect_ratio < 13):
                    area = length * height
                    print("area "+str(area))
                    if (area > 400 and area < 4700):

                        x1 = min(point1[0], point2[0])
                        y1 = min(point1[1], point2[1])
                        x2 = max(point1[0], point2[0])
                        y2 = max(point1[1], point2[1])
                        sum_sat = integral_image_sat[y2 + 1, x2 + 1] + integral_image_sat[y1, x1] - \
                                  integral_image_sat[y2 + 1, x1] - integral_image_sat[y1, x2 + 1]
                        mean_sat = sum_sat / area
                        print("mean sat "+str(mean_sat))
                        if (mean_sat < 140):

                            sum_val = integral_image_val[y2 + 1, x2 + 1] + integral_image_val[y1, x1] - \
                                      integral_image_val[y2 + 1, x1] - integral_image_val[y1, x2 + 1]
                            mean_val = sum_val / area
                            print("mean val " + str(mean_val))
                            if (mean_val < 130):  # dark background -> white letters
                                target_integral_image = integral_threshwhite_big
                            if (mean_val > 130):  # bright background -> dark letters
                                target_integral_image = integral_threshblack_big
                            x1_big, y1_big, x2_big, y2_big = int(x1 / ratio), int(y1 / ratio), int(x2 / ratio), int(
                                y2 / ratio)
                            count_letters_big = target_integral_image[y2_big + 1, x2_big + 1] + target_integral_image[
                                y1_big, x1_big] - \
                                                target_integral_image[y2_big + 1, x1_big] - target_integral_image[
                                                    y1_big, x2_big + 1]
                            area_big = area / ratio / ratio
                            filling_letters = count_letters_big / area_big
                            print("filling_letters " + str(filling_letters))
                            if (filling_letters > 0.02 and filling_letters < 0.35):
                                print("TEST GT PASSED")
                                cv2.rectangle(rgb_image, pt1=point1, pt2=point2, color=(0,255,0), thickness=2)




        count = 0
        window_candidates = []
        for point1 in candidate_points:
            for point2 in candidate_points:

                ######################################## FILTER WINDOWS ###############################################
                length = abs(point1[0]-point2[0])
                if(length >50 and length < 160):

                    height = abs(point1[1] - point2[1])
                    if(height > 0):

                        aspect_ratio = length / abs(point1[1] - point2[1])
                        if (aspect_ratio>4 and aspect_ratio<13):

                            area = length*height
                            if(area > 400 and area < 4700):

                                x1 = min(point1[0], point2[0])
                                y1 = min(point1[1], point2[1])
                                x2 = max(point1[0], point2[0])
                                y2 = max(point1[1], point2[1])
                                sum_sat = integral_image_sat[y2 + 1, x2 + 1] + integral_image_sat[y1, x1] - \
                                          integral_image_sat[y2 + 1, x1] - integral_image_sat[y1, x2 + 1]
                                mean_sat = sum_sat / area

                                if(mean_sat < 140):

                                    sum_val = integral_image_val[y2 + 1, x2 + 1] + integral_image_val[y1, x1] - \
                                              integral_image_val[y2 + 1, x1] - integral_image_val[y1, x2 + 1]
                                    mean_val = sum_val / area
                                    if (mean_val < 130): #dark background -> white letters
                                        target_integral_image = integral_threshwhite_big
                                    if (mean_val > 130): #bright background -> dark letters
                                        target_integral_image = integral_threshblack_big
                                    x1_big, y1_big, x2_big, y2_big = int(x1/ratio), int(y1/ratio), int(x2/ratio), int(y2/ratio)
                                    count_letters_big = target_integral_image[y2_big + 1, x2_big + 1] + target_integral_image[y1_big, x1_big] - \
                                                          target_integral_image[y2_big + 1, x1_big] - target_integral_image[y1_big, x2_big + 1]
                                    area_big = area/ratio/ratio
                                    filling_letters = count_letters_big/area_big
                                    if(filling_letters > 0.05 and filling_letters < 0.35):
                                        count+=1
                                        #cv2.rectangle(rgb_image, pt1=point1, pt2=point2, color=(0,0,255), thickness=1)

                                        #################### SCORE WINDOWS AND RETRIEVE BEST ##########################

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

                                        ################### Score and distance ##############################

                                        distance = abs(aspect_ratio-mean_aspect_ratio)/std_aspect_ratio + \
                                            abs(length-mean_length)/std_length + \
                                            abs(area-mean_area)/std_area + \
                                            abs(filling_letters-mean_filling_ratio)/std_fillin_ratio + \
                                            abs(mean_sat-mean_saturation)/std_saturation

                                        x2 = min(x2, im.shape[1] - 2)
                                        y2 = min(y2, im.shape[0] - 2)

                                        count_gradient_htop = (integral_mix_sobely[y1 + 2, x2 + 1] + \
                                                              integral_mix_sobely[y1-1, x1] - \
                                                              integral_mix_sobely[y1-1 , x2+1] - \
                                                              integral_mix_sobely[y1+2, x1]) / ((x2-x1)*3)

                                        count_gradient_hbot = (integral_mix_sobely[y2 + 2, x2 + 1] + \
                                                              integral_mix_sobely[y2 - 1, x1] - \
                                                              integral_mix_sobely[y2 - 1, x2 + 1] - \
                                                              integral_mix_sobely[y2 + 2, x1]) / ((x2-x1)*3)

                                        count_gradient_vleft = (integral_mix_sobely[y2 + 1, x1 + 2] + \
                                                              integral_mix_sobely[y1, x1-1] - \
                                                              integral_mix_sobely[y1, x1 + 2] - \
                                                              integral_mix_sobely[y2 + 1, x1-1]) / ((y2-y1)*3)

                                        count_gradient_right = (integral_mix_sobely[y2 + 1, x2 + 2] + \
                                                               integral_mix_sobely[y1, x2 - 1] - \
                                                               integral_mix_sobely[y1, x2 + 2] - \
                                                               integral_mix_sobely[y2 + 1, x2 - 1]) / ((y2-y1)*3)

                                        gradient_score = count_gradient_htop + count_gradient_hbot + count_gradient_vleft + count_gradient_right

                                        print("-----------------------")
                                        print(distance)
                                        print(gradient_score)

                                        window_candidates.append(((x1, y1, x2, y2), distance-3*gradient_score))








        print("candidate windows")
        print(count)
        winning_window = min(window_candidates, key=lambda x: x[1])
        print("WINNER")
        print(winning_window)
        p1 = winning_window[0][:2]
        p2 = winning_window[0][2:]
        print(p1)
        print(p2)
        cv2.rectangle(rgb_image, pt1=p1, pt2=p2, color=(0, 0, 255), thickness=2)
        print("-----------------------------")

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

        """






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

        """plt.subplot(3, 4, 8)
        plt.title("Edges")
        plt.imshow(edges, cmap='gray')

        plt.subplot(3, 4, 9)
        plt.title("Corners Sat")
        plt.imshow(sat_corners, cmap='gray')

        plt.subplot(3, 4, 10)
        plt.title("Corners Val")
        plt.imshow(val_corners, cmap='gray')
        """
        plt.show()




if __name__ == "__main__":
    main()