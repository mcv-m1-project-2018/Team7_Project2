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
from descriptors.orb import Orb


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


def get_features(method, data, query_folder, database_dir, method_name):
    """
    Computes or loads the features. The first time has to compute all the features and adapt the threshold, but once
    computed and stored it's faster. This function has to check for the files for every query and week, so it's kind
    of messy.
    :param method:  method object (surf or orb)
    :param data:  data
    :param query_folder: used to name the pkl feature files
    :param database_dir: db directory
    :param method_name: method name (string surf or orb)
    :return: database features, query features
    """
    # compute the features just once. we only keep the descriptors (pickling cv2 keypoints is a nightmare, but we don't
    # use them anyway)

    if not os.path.isdir("features"):
        os.mkdir("features")

    # db features
    db_name = "features/" + method_name + "_features_" + database_dir + ".pkl"
    if os.path.isfile(db_name):
        print("loading " + database_dir + " features...")
        with open(db_name, "rb") as f:
            database_feats = pickle.load(f)
    else:
        print("computing " + database_dir + " features...")
        database_feats = {}
        for im, name in data.database_imgs:
            if method_name == 'surf':
                database_feats[name] = method.detectAndCompute(im, adapt_threshold=True, step=1000, max_features=1000)
            if method_name == 'orb':
                database_feats[name] = method.detectAndCompute(im)
        with open(db_name, "wb") as f:
            pickle.dump(database_feats, f)

    # query features
    query_name = "features/" + method_name + "_features_" + query_folder + ".pkl"
    if os.path.isfile(query_name):
        print("loading " + query_folder + " features...")
        with open(query_name, "rb") as f:
                query_feats = pickle.load(f)
    else:
        query_feats = {}
        print("computing " + query_folder + " features...")
        for im, name in data.query_imgs:
            if method_name == 'surf':
                query_feats[name] = method.detectAndCompute(im, adapt_threshold=True, step=1000, max_features=1000)
            if method_name == 'orb':
                query_feats[name] = method.detectAndCompute(im)
        with open(query_name, "wb") as f:
            pickle.dump(query_feats, f)

    return database_feats, query_feats


def main(database_dir, query_folder, ground_truth, method_name):
    if method_name == 'orb':
        method = Orb()
        min_features = 25
    else:
        method = Surf()
        min_features = 100

    data = Data(database_dir=database_dir, query_dir=query_folder)

    query_imgs = [[im, name] for im, name in data.query_imgs]
    database_imgs = [[im, name] for im, name in data.database_imgs]
    database_feats, query_feats = get_features(method, data, query_folder, database_dir, method_name)

    eval_array = []
    for q_im, q_name in query_imgs:
        scores = method.retrieve_best_results(q_im, database_imgs, database_feats, query_feats[q_name])

        if not args.week3:  # week 4 evaluation
            if scores[0][1] < min_features:  # minimum number of features matched allowed (-1 otherwise)
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
    parser.add_argument('-use_orb', help='uses ORB to match the images',
                        action='store_true')
    args = parser.parse_args()

    # here we select the folder with the right database and queries according to the options
    if args.week3:
        database_dir = 'museum_set_random'
    else:
        database_dir = 'BBDD_W4'

    if args.test and args.week3:                # week 3 test set
        query_folder = "query_test_random"
        ground_truth = ground_truth_test
    elif not args.test and args.week3:          # week 3 dev set
        query_folder = "query_devel_random"
        ground_truth = ground_truth_val
    elif args.test and not args.week3:          # week 4 test set
        query_folder = None
        ground_truth = None
        print("no test set for week4 yet")
        exit(0)
    else:                                       # week 4 dev set
        query_folder = "query_devel_W4"
        ground_truth = ground_truth_W4

    if args.use_orb:
        method_name = 'orb'
    else:
        method_name = 'surf'

    main(database_dir, query_folder, ground_truth, method_name)