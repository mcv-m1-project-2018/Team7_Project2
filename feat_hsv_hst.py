import cv2
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance



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


def compare_histograms(hist1, hist2, method):
    score = 0
    if method == cv2.HISTCMP_INTERSECT or method == cv2.HISTCMP_CHISQR or method == cv2.HISTCMP_CORREL or \
            method == cv2.HISTCMP_BHATTACHARYYA or method == cv2.HISTCMP_KL_DIV or method == cv2.HISTCMP_INTERSECT:
        score = cv2.compareHist(hist1, hist2, method)

    if method == 'emd':
        score = wasserstein_distance(hist1[:, 0], hist2[:, 0])  # earth mover's distance

    return score


def retrieve_best_results(image_histH, data, method=cv2.HISTCMP_BHATTACHARYYA, K=10):
    """
    Call this function in order to compare a histogram with the rest of the dataset
    :param image_histH: hsv histogram of the image to match
    :param data: the dataset
    :param method:
    :param K:
    :return: the k top matches
    """

    scores = []
    for database_im, database_im_name in data.database_imgs:
        hsv_hist = get_hsv(database_im)
        scores.append((database_im_name, compare_histograms(image_histH, hsv_hist['histH'], method=method)))

    if method in [cv2.HISTCMP_INTERSECT, cv2.HISTCMP_CORREL]:
        scores.sort(key=lambda s: s[1], reverse=True)
    else:
        scores.sort(key=lambda s: s[1], reverse=False)

    return scores[:K]

