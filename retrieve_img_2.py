import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from descriptors.feat_hsv_hst import retrieve_best_results as retrieve_best_results_hsv, get_hsv_hist

from data_handler import Data
from dataset_w1_gt import ground_truth as ground_truth_val, ground_truth_test
from descriptors.feat_wavelet_hash import retrieve_best_results, get_hash
from evaluation import evaluation


def show_results(scores, museum_set, query_image, ground_truth_image):
    RGB_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    plt.subplot(353), plt.imshow(RGB_image)
    RGB_image_gt = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2RGB)
    plt.subplot(354), plt.imshow(RGB_image)
    for i in range(10):
        RGB_image = cv2.cvtColor(museum_set[scores[i][0]]['image'], cv2.COLOR_BGR2RGB)
        plt.subplot(3, 5, 6 + i), plt.imshow(RGB_image)
    plt.show()


def main(args):
    if args.test:
        query_folder = "query_test_random"
        ground_truth = ground_truth_test
    else:
        query_folder = "query_devel_random"
        ground_truth = ground_truth_val

    data = Data(database_dir='museum_set_random', query_dir=query_folder)

    # test_ground_truth(ground_truth=ground_truth, museum_set=museum_set, query_set=query_set)
    eval_array = []

    query_imgs = [[im, name] for im, name in data.query_imgs]
    database_imgs = [[im, name] for im, name in data.database_imgs]
    query_hist = [get_hsv_hist(im) for im, name in query_imgs]
    database_hist = [get_hsv_hist(im) for im, name in database_imgs]

    database_hash = {}
    for im, name in data.database_imgs:
        database_hash[name] = get_hash(im)

    for [q_image, q_name], q_hist in zip(query_imgs, query_hist):
        if args.use_histogram:
            K = len(database_imgs)
        else:
            K = 10

        scores = retrieve_best_results(q_image, database_imgs, database_hash, K=K)

        if args.use_histogram:
            scores2 = retrieve_best_results_hsv(image_histH=q_hist,
                                                database_imgs=database_imgs,
                                                database_hist=database_hist, K=K)
            # sort by image name
            scores.sort(key=lambda s: s[0], reverse=False)
            scores2.sort(key=lambda s: s[0], reverse=False)

            # add the scores (assuming we are using cv2.HISTCMP_BHATTACHARYYA, as it outputs the best match as the
            # lowest score)
            combined_scores = [(score[0][0], score[1][1]+score[0][1]) for score in zip(scores, scores2)]

            combined_scores.sort(key=lambda s: s[1], reverse=False)
            combined_scores = combined_scores[:10]
            scores = combined_scores

        eval = evaluation(predicted=[s[0] for s in scores], actual=[ground_truth[q_name]])
        print(eval)
        eval_array.append(eval)
        """
        show_results(scores=scores,
                     museum_set=museum_set,
                     query_image=query_set[query]['image'],
                     ground_truth_image=data.database_imgs.read( ground_truth[query_name] ))
        """
    global_eval = np.mean(eval_array)
    print("----------------\nEvaluation: " + str(global_eval))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-use_histogram', help='use the hashing method with the histogram based method',
                        action='store_true')
    parser.add_argument('-test', help='runs the code on the test dataset',
                        action='store_true')
    args = parser.parse_args()
    main(args)

