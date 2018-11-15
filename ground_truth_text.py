from data_handler import Data
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
def get_text_gt():
    gt_list = [(95, 867, 767, 991), (130, 588, 984, 740), (70, 489, 395, 552), (128, 1072, 649, 1188),
               (149, 868, 1114, 1044), (160, 763, 753, 879), (92, 629, 272, 673), (149, 55, 584, 140),
               (199, 978, 825, 1094), (227, 183, 1062, 328), (131, 116, 1076, 281), (121, 0, 876, 145),
               (124, 37, 493, 122), (151, 133, 1096, 298), (91, 480, 450, 565), (192, 172, 895, 317),
               (50, 59, 375, 122), (134, 140, 529, 225), (151, 5, 891, 87), (159, 9, 1134, 110), (165, 3, 435, 47),
               (276, 811, 947, 896), (223, 862, 1059, 978), (287, 71, 970, 156), (160, 26, 795, 107),
               (121, 972, 804, 1057), (233, 960, 870, 1045), (117, 575, 414, 631), (140, 996, 823, 1081),
               (139, 709, 498, 765), (183, 50, 1121, 166), (247, 692, 930, 777), (276, 89, 947, 174),
               (137, 695, 1051, 811), (108, 1, 729, 86), (174, 725, 502, 781), (179, 521, 476, 577),
               (170, 988, 729, 1073), (119, 860, 421, 916), (193, 151, 762, 236), (130, 583, 515, 639),
               (288, 225, 711, 281), (174, 42, 746, 110), (152, 127, 1082, 223), (181, 1133, 850, 1218),
               (238, 698, 273, 723), (177, 81, 928, 162), (162, 563, 504, 619), (87, 578, 482, 634),
               (210, 116, 595, 172), (212, 105, 607, 161), (202, 35, 937, 120), (138, 49, 556, 105),
               (115, 43, 915, 123), (239, 107, 274, 132), (128, 1099, 797, 1184), (173, 1053, 745, 1121),
               (155, 733, 467, 777), (133, 534, 934, 619), (157, 77, 1023, 159), (196, 138, 1062, 220),
               (122, 848, 694, 916), (57, 497, 475, 553), (142, 452, 494, 508), (78, 634, 501, 690),
               (261, 80, 930, 165), (108, 469, 420, 513), (121, 504, 506, 560), (202, 80, 1068, 162),
               (225, 817, 643, 873), (135, 3, 553, 59), (270, 84, 688, 140), (166, 782, 1096, 878),
               (156, 756, 745, 819), (94, 534, 743, 619), (142, 139, 893, 220), (138, 79, 531, 132),
               (234, 704, 903, 789), (152, 649, 903, 730), (213, 59, 936, 144), (223, 656, 956, 741),
               (257, 476, 1243, 592), (117, 100, 401, 144), (196, 655, 919, 740), (166, 53, 545, 109),
               (100, 489, 412, 545), (192, 635, 781, 720), (96, 30, 460, 86), (209, 516, 1195, 632),
               (276, 31, 1262, 147), (277, 68, 970, 153), (154, 206, 823, 291), (54, 71, 413, 124),
               (158, 159, 891, 244), (218, 27, 253, 52), (128, 913, 661, 976), (131, 495, 664, 558),
               (180, 155, 1202, 271), (206, 600, 1192, 716), (208, 85, 542, 141), (154, 191, 619, 254),
               (147, 849, 397, 893), (178, 864, 1165, 988), (92, 24, 421, 80), (68, 33, 366, 89),
               (150, 671, 1045, 772), (45, 36, 379, 92), (133, 314, 383, 358), (87, 163, 710, 248),
               (137, 877, 988, 986), (208, 25, 1195, 170), (114, 874, 1009, 975), (153, 615, 730, 700),
               (255, 232, 1335, 408), (356, 2229, 2740, 2525), (281, 2162, 2143, 2398), (278, 486, 2433, 751),
               (255, 12, 2210, 337), (345, 2148, 2468, 2413), (344, 53, 2339, 318), (408, 2181, 3623, 2566),
               (362, 10, 2829, 302), (411, 2334, 2854, 2625), (354, 260, 2453, 513), (319, 2022, 2794, 2347),
               (296, 2382, 2567, 2654), (380, 2960, 2367, 3225), (317, 170, 2160, 435), (308, 2440, 2295, 2705),
               (373, 300, 2816, 591), (350, 65, 2863, 385), (389, 39, 2468, 275), (368, 374, 2551, 579),
               (398, 2133, 2773, 2329), (231, 565, 2018, 770), (324, 394, 2297, 599), (350, 102, 2646, 338),
               (264, 500, 2119, 658), (252, 362, 2225, 567), (334, 500, 1929, 639), (302, 245, 2329, 450),
               (367, 218, 2726, 454), (372, 425, 2129, 630), (292, 2436, 2265, 2641), (419, 40, 3324, 365),
               (327, 2034, 2406, 2270), (412, 1737, 3075, 2033), (251, 2161, 2254, 2354), (437, 116, 3332, 350),
               (450, 1881, 2809, 2117), (408, 15, 2591, 220), (412, 413, 2511, 678), (291, 2091, 2390, 2356),
               (393, 2174, 2768, 2499), (349, 340, 2246, 576), (281, 2272, 2416, 2487), (269, 2422, 1900, 2615),
               (271, 3219, 2406, 3434), (390, 2169, 3233, 2554), (176, 21, 1501, 197), (441, 25, 3050, 381),
               (343, 1945, 2700, 2241), (437, 58, 2812, 383), (294, 60, 2625, 325), (484, 1892, 3179, 2217),
               (340, 317, 2097, 522), (345, 2471, 2486, 2767), (391, 2431, 2288, 2667), (254, 2693, 2011, 2898),
               (182, 2545, 991, 2661), (379, 1796, 2486, 2061), (242, 2277, 1807, 2442), (368, 310, 2698, 546),
               (368, 176, 3004, 440), (341, 38, 2320, 243), (324, 2590, 2183, 2795), (352, 1340, 2515, 1576),
               (396, 1909, 2762, 2205), (405, 2636, 2519, 2872), (427, 1471, 3135, 1767), (317, 534, 2638, 830),
               (361, 1810, 2986, 2135), (244, 255, 1833, 460), (336, 2056, 2315, 2261), (269, 1975, 1216, 2091),
               (296, 3304, 2167, 3497), (348, 2846, 2165, 3051), (311, 4, 2607, 240), (470, 82, 3089, 347),
               (394, 2981, 2367, 3186), (445, 41, 3273, 277), (261, 309, 2294, 514), (369, 484, 2284, 660),
               (336, 1591, 2632, 1827), (370, 37, 2661, 242), (387, 61, 2420, 266), (296, 507, 2527, 712),
               (254, 205, 1734, 325), (401, 572, 2374, 777), (407, 2335, 2773, 2571), (396, 2015, 2451, 2191),
               (394, 433, 2367, 638), (250, 2499, 2019, 2638), (434, 1555, 2467, 1760), (324, 2105, 2790, 2313)]

    gt_dict = {}
    for id in range(len(gt_list)):
        im_name = 'ima_'+'{:06d}'.format(id)
        gt_dict[im_name] = gt_list[id]

    return gt_dict

def analyze_gt(data, gt):
    aspect_ratios = []
    lengths = []
    heights = []
    areas = []
    values = []
    saturations = []
    filling_letters = []
    for im, im_name in data.database_imgs:

        hsv_image_big = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        threshblack_big = cv2.inRange(hsv_image_big, (0, 0, 0), (180, 50, 50))/255
        threshwhite_big = cv2.inRange(hsv_image_big, (0, 0, 200), (180, 50, 255))/255

        integral_threshblack_big = cv2.integral(threshblack_big)
        integral_threshwhite_big = cv2.integral(threshwhite_big)



        x1, y1, x2, y2 = gt[im_name]
        aspect_ratios.append((x2-x1)/(y2-y1))
        FinalSize = 200

        shape_max = max(im.shape)
        ratio = FinalSize / shape_max

        im = cv2.resize(im, (0, 0), fx=ratio, fy=ratio)
        maxY, maxX = im.shape[:2]
        x1 = min(int(x1 * ratio), maxX-1)
        y1 = min(int(y1 * ratio), maxY-1)
        x2 = min(int(x2 * ratio), maxX-1)
        y2 = min(int(y2 * ratio), maxY-1)
        lengths.append(x2-x1)
        heights.append(y2-y1)
        area = (x2-x1)*(y2-y1)
        areas.append(area)

        hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        s = hsv_image[:,:,1]
        v = hsv_image[:, :, 2]
        integral_image_sat = cv2.integral(s)
        integral_image_val = cv2.integral(v)

        sum_sat = integral_image_sat[y2+1,x2+1] + integral_image_sat[y1,x1] - integral_image_sat[y2+1,x1] - integral_image_sat[y1, x2+1]
        mean_sat = sum_sat / area

        sum_val = integral_image_val[y2+1, x2+1] + integral_image_val[y1, x1] - integral_image_val[y2+1, x1] - integral_image_val[y1, x2+1]
        mean_val = sum_val / area

        values.append(mean_val)
        saturations.append(mean_sat)

        if (mean_val < 130):  # dark background -> white letters
            target_integral_image = integral_threshwhite_big
        if (mean_val > 130):  # bright background -> dark letters
            target_integral_image = integral_threshblack_big
        x1_big, y1_big, x2_big, y2_big = int(x1 / ratio), int(y1 / ratio), int(x2 / ratio), int(y2 / ratio)
        count_letters_big = target_integral_image[y2_big + 1, x2_big + 1] + target_integral_image[y1_big, x1_big] - \
                            target_integral_image[y2_big + 1, x1_big] - target_integral_image[y1_big, x2_big + 1]
        area_big = (x2_big-x1_big)*(y2_big-y1_big)
        filling_letter = count_letters_big / area_big
        filling_letters.append(filling_letter)



        """if (filling_letter < 0.01):
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plt.subplot(131)
            plt.imshow(im)
            plt.subplot(132)
            plt.imshow(s, cmap='gray')
            plt.subplot(133)
            plt.imshow(integral_image_sat)
            plt.show()
        """

    print("min aspect ratio "+str(min(aspect_ratios)))
    print("max aspect ratio " + str(max(aspect_ratios)))
    print("mean aspect ratio " + str(np.mean(aspect_ratios)))
    print("std aspect ratio " + str(np.std(aspect_ratios)))

    print("min length " + str(min(lengths)) + "    ("+str(FinalSize)+")")
    print("max length " + str(max(lengths)) + "    ("+str(FinalSize)+")")
    print("mean length " + str(np.mean(lengths)) + "    (" + str(FinalSize) + ")")
    print("std length " + str(np.std(lengths)) + "    (" + str(FinalSize) + ")")

    print("min height " + str(min(heights)) + "    (" + str(FinalSize) + ")")
    print("max height " + str(max(heights)) + "    (" + str(FinalSize) + ")")
    print("mean height " + str(np.mean(heights)) + "    (" + str(FinalSize) + ")")
    print("std height " + str(np.std(heights)) + "    (" + str(FinalSize) + ")")

    print("min area " + str(min(areas)) + "    (" + str(FinalSize) + ")")
    print("max area " + str(max(areas)) + "    (" + str(FinalSize) + ")")
    print("mean are " + str(np.mean(areas)) + "    (" + str(FinalSize) + ")")
    print("std area " + str(np.std(areas)) + "    (" + str(FinalSize) + ")")

    print("min filling ratio lettes " + str(min(filling_letters)))
    print("max filling ratio letters " + str(max(filling_letters)))
    print("mean filling ratio letters " + str(np.mean(filling_letters)))
    print("std filling ratio letters " + str(np.std(filling_letters)))

    print("min saturation " + str(min(saturations)))
    print("max saturation " + str(max(saturations)))
    print("mean saturation " + str(np.mean(saturations)))
    print("std saturation " + str(np.std(saturations)))



    plt.subplot(341)
    plt.title("Aspect Ratio")
    plt.hist(aspect_ratios, bins=100)
    plt.subplot(342)
    plt.title("Length"+ "    ("+str(FinalSize)+")")
    plt.hist(lengths, bins=100)
    plt.subplot(343)
    plt.title("Area " + "    (" + str(FinalSize) + ")")
    plt.hist(areas, bins=100)
    plt.subplot(344)
    plt.title("Length and Aspect Ratio"+ "    ("+str(FinalSize)+")")
    plt.hist2d(lengths, aspect_ratios)
    plt.subplot(345)
    plt.title("Length and Area" + "    (" + str(FinalSize) + ")")
    plt.hist2d(lengths, areas)
    plt.subplot(346)
    plt.title("Area and Aspect Ratio" + "    (" + str(FinalSize) + ")")
    plt.hist2d(areas, aspect_ratios)
    plt.subplot(347)
    plt.title("Mean Saturation")
    plt.hist(saturations, bins=100)
    plt.subplot(348)
    plt.title("Mean Value")
    plt.hist(values, bins=100)
    plt.subplot(349)
    plt.title("Mean Saturation and Value")
    plt.hist2d(saturations, values)
    plt.subplot(3,4,10)
    plt.title("Filling Letters")
    plt.hist(filling_letters, bins=100)
    plt.show()



def test_gt(data, gt):
    # loop over database_imgs without overloading memory
    for im, im_name in data.database_imgs:
        FinalSize = 200
        shape_max = max(im.shape)
        ratio = FinalSize / shape_max
        im = cv2.resize(im, (0, 0), fx=ratio, fy=ratio)
        x1, y1, x2, y2 = gt[im_name]
        cv2.rectangle(im, (int(x1*ratio), int(y1*ratio)), (int(x2*ratio), int(y2*ratio)), (0, 255, 0), 2)
        plt.imshow(im)
        plt.show()






def main():
    data = Data(database_dir='w5_BBDD_random', query_dir='w5_devel_random')
    gt = get_text_gt()
    #print(gt)
    #test_gt(data, gt)
    analyze_gt(data, gt)

if __name__ == "__main__":
    main()