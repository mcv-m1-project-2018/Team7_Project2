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

        hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        value_image = hsv_image[:,:,2]

        x1, y1, x2, y2 = gt[im_name]

        threshblack = cv2.inRange(hsv_image, (0,0,0), (180, 50, 50))
        threshwhite = cv2.inRange(hsv_image, (0, 0, 200), (180, 50, 255))

        cv2.rectangle(threshblack, (x1, y1), (x2,y2), (150), 10)
        cv2.rectangle(threshwhite, (x1, y1), (x2, y2), (150), 10)

        integral_threshblack = cv2.integral(threshblack)/255
        integral_threshwhite = cv2.integral(threshwhite)/255

        integral_image_val = cv2.integral(value_image)
        sum_val = integral_image_val[y2 + 1, x2 + 1] + integral_image_val[y1, x1] - integral_image_val[y2 + 1, x1] - \
                  integral_image_val[y1, x2 + 1]

        area = (x2 - x1) * (y2 - y1)
        mean_val = sum_val / area
        if (mean_val < 130):  # dark background -> white letters
            target_integral_image = integral_threshwhite
            print("Dark background")
        if (mean_val > 130):  # bright background -> dark letters
            target_integral_image = integral_threshblack
            print("White background")
        count_letters = target_integral_image[y2 + 1, x2 + 1] + target_integral_image[y1, x1] - \
                            target_integral_image[y2 + 1, x1] - target_integral_image[y1, x2 + 1]
        filling_letter = count_letters / area
        print("Filling letter:" + str(filling_letter))


        plt.subplot(131)
        plt.title("Image")
        plt.imshow(im)
        plt.subplot(132)
        plt.title("Black Segmentation")
        plt.imshow(threshblack, cmap='gray')
        plt.subplot(133)
        plt.title("White Segmentation")
        plt.imshow(threshwhite, cmap='gray')


        plt.show()









if __name__ == "__main__":
    main()