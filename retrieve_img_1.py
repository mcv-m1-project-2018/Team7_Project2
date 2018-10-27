import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataset_w1_gt import ground_truth
from feat_hsv_hst import get_hsv
from data_handler import Data

def difference(image1_histH, image2_histH, metric=cv2.HISTCMP_CORREL):
    return cv2.compareHist(image1_histH, image2_histH, metric)

def retrieve_best_results(image_histH, data, K=10):
    scores = []
    for database_im, database_im_name in data.database_imgs:
        hsv_hist = get_hsv( database_im )
        scores.append( ( database_im_name , difference(image_histH, hsv_hist['histH'])) )

    scores.sort(key=lambda s: s[1], reverse=True)
    return scores[:K]


def show_results(scores, museum_set, query_image, ground_truth_image):
    RGB_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    plt.subplot(353), plt.imshow(RGB_image)
    RGB_image_gt = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2RGB)
    plt.subplot(354), plt.imshow(RGB_image)
    for i in range(10):
        RGB_image = cv2.cvtColor(museum_set[scores[i][0]]['image'], cv2.COLOR_BGR2RGB)
        plt.subplot(3,5,6+i), plt.imshow(RGB_image)
    plt.show()


def evaluation(predicted, actual, k=10):
    score    = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def main():

    data = Data(database_dir= 'museum_set_random', query_dir= 'query_devel_random')
    # test_ground_truth(ground_truth=ground_truth, museum_set=museum_set, query_set=query_set)
    eval_array = []
    for query_im, query_name in data.query_imgs:
        hsv_hist = get_hsv( query_im )
        scores = retrieve_best_results(image_histH=hsv_hist['histH'], data = data)

        eval = evaluation(predicted=[s[0] for s in scores], actual=[ground_truth[query_name]])
        print(eval)
        eval_array.append(eval)
        """
        show_results(scores=scores,
                     museum_set=museum_set,
                     query_image=query_set[query]['image'],
                     ground_truth_image=data.database_imgs.read( ground_truth[query_name] ))
        """
    global_eval = np.mean(eval_array)
    print("----------------\nEvaluation: "+str(global_eval))



if __name__ == "__main__":
    main()