import os
import pickle
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from data_handler import Data
from dataset_w1_gt import ground_truth as ground_truth_val, ground_truth_test, ground_truth_W4
from evaluation import evaluation
from descriptors.surf import Surf


def show_results(scores, data, query_image, ground_truth_image, display=True):
    if display:
        RGB_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        plt.subplot(353), plt.imshow(RGB_image)

        RGB_image_gt = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2RGB)
        plt.subplot(354), plt.imshow(RGB_image)

        for i in range(10):
            RGB_image = cv2.cvtColor(data.database_imgs.read(scores[i][0] + '.jpg'), cv2.COLOR_BGR2RGB)
            plt.subplot(3, 5, 6 + i), plt.imshow(RGB_image)
        plt.show()


def get_features(surf, data, query_folder, database_dir):
    """
    Computes or loads the features. The first time has to compute all the features and adapt the threshold, but once
    computed and stored it's faster.
    :param surf:  surf object
    :param data:  data
    :param query_folder: used to name the pkl feature files
    :param database_dir: db directory
    :return: database features, query features
    """
    # compute the features just once. we only keep the descriptors (pickling cv2 keypoints is a nightmare, but we don't
    # use them anyway)
    surf.set_hessian_threshold(350)

    if not os.path.isdir("features"):
        os.mkdir("features")

    # db features
    if os.path.isfile("features/surf_features_" + database_dir + ".pkl"):
        print("loading " + database_dir + " features...")
        with open("features/surf_features_" + database_dir + ".pkl", "rb") as f:
            database_feats = pickle.load(f)
    else:
        print("computing " + database_dir + " features...")
        database_feats = {}
        for im, name in data.database_imgs:
            database_feats[name] = surf.detectAndCompute(im, adapt_threshold=True, step=1000, max_features=1000)
        with open("features/surf_features_" + database_dir + ".pkl", "wb") as f:
            pickle.dump(database_feats, f)

    # query features
    if os.path.isfile("features/surf_features_" + query_folder + ".pkl"):
        print("loading " + query_folder + " features...")
        with open("features/surf_features_" + query_folder + ".pkl", "rb") as f:
                query_feats = pickle.load(f)
    else:
        query_feats = {}
        print("computing " + query_folder + " features...")
        for im, name in data.query_imgs:
            query_feats[name] = surf.detectAndCompute(im, adapt_threshold=True, step=1000, max_features=1000)
        with open("features/surf_features_" + query_folder + ".pkl", "wb") as f:
            pickle.dump(query_feats, f)

    return database_feats, query_feats


def main(args):
    # here we select the folder with the right database and queries according to the options
    if args.test and args.week3:                # week 3 test set
        query_folder = "query_test_random"
        database_dir = 'museum_set_random'
        ground_truth = ground_truth_test
    elif not args.test and args.week3:          # week 3 dev set
        query_folder = "query_devel_random"
        database_dir = 'museum_set_random'
        ground_truth = ground_truth_val
    elif args.test and not args.week3:          # week 4 test set
        query_folder = None
        ground_truth = None
        database_dir = 'BBDD_W4'
        print("no test set for week4 yet")
        exit(0)
    else:                                       # week 4 dev set
        query_folder = "query_devel_W4"
        database_dir = 'BBDD_W4'
        ground_truth = ground_truth_W4

    data = Data(database_dir=database_dir, query_dir=query_folder)

    eval_array = []
    surf = Surf()  # Surf object (from descriptors folder)

    query_imgs = [[im, name] for im, name in data.query_imgs]
    database_imgs = [[im, name] for im, name in data.database_imgs]

    # here the features are created/loaded
    database_feats, query_feats = get_features(surf, data, query_folder, database_dir)

    for q_im, q_name in query_imgs:
        scores = surf.retrieve_best_results(q_im, database_imgs, database_feats, query_feats[q_name])

        if not args.week3:  # week 4 evaluation
            # if there are not enough matches set the scores to be a list with only -1 and 0 score
            if scores[0][1] < 100:
                scores = [(-1, 0)]
            eval = evaluation(predicted=[s[0] for s in scores], actual=ground_truth[q_name])
        else:  # week 3 evaluation
            eval = evaluation(predicted=[s[0] for s in scores], actual=[ground_truth[q_name]])

        eval_array.append(eval)
        print(eval)

    global_eval = np.mean(eval_array)
    print("----------------\nEvaluation: " + str(global_eval))

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', help='runs the code on the test dataset (only week3 by now)',
                        action='store_true')
    parser.add_argument('-week3', help='uses the queries and the data from the week 3',
                        action='store_true')
    args = parser.parse_args()
    main(args)
