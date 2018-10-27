import cv2
import matplotlib.pyplot as plt



def get_hsv(img, visualize=False):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    histH = cv2.calcHist(images=[hsv_image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    histS = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    histV = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
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
