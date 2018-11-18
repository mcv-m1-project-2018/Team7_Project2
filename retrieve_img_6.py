import os
import pickle
import argparse

import cv2
import numpy as np

from data_handler import Data
from dataset_w1_gt import ground_truth_W5
from evaluation import evaluation
from descriptors.surf import Surf
from descriptors.orb import Orb
from descriptors.sift import Sift
from descriptors.root_sift import RootSift
from painting_detection import get_painting_rotated
from detect_textbb_2 import get_text_bbox


def reshape(image):
    shape_max = max(image.shape)
    if shape_max > 750:  # some of the images are HUGE
        ratio = 750 / shape_max
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
    return image


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
            x, y, w, h = get_text_bbox(im)
            im[y:h, x:w, :] = 0
            im = reshape(im)
            if method_name == 'surf':
                database_feats[name] = method.detectAndCompute(im, adapt_threshold=True, step=1000, max_features=2000)
            if method_name == 'orb':
                database_feats[name] = method.detectAndCompute(im)
            if method_name == 'sift' or method_name == 'root_sift':
                kps, descriptors = method.detectAndCompute(im)
                features = []
                if len(kps):
                    for kp, desc in zip(kps, descriptors):
                        temp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, desc)
                        features.append(temp)
                database_feats[name] = features
        with open(db_name, "wb") as f:
            pickle.dump(database_feats, f)

    if method_name == 'sift' or method_name == 'root_sift':
        unserialized = {}
        for name in database_feats:
            unserialized[name] = [[], []]
            for f in database_feats[name]:
                kp = cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=f[1], _angle=f[2],
                                  _response=f[3], _octave=f[4], _class_id=f[5])
                desc = f[6]
                unserialized[name][0].append(kp)
                unserialized[name][1].append(desc)
            if len(unserialized[name][1]):
                unserialized[name][1] = np.vstack(unserialized[name][1])
        database_feats = unserialized

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
            im, _ = get_painting_rotated(im)
            if method_name == 'surf':
                query_feats[name] = method.detectAndCompute(im, adapt_threshold=True, step=1000, max_features=2000)
            if method_name == 'orb':
                query_feats[name] = method.detectAndCompute(im)
            if method_name == 'sift' or method_name == 'root_sift':
                kps, descriptors = method.detectAndCompute(im)
                features = []
                if len(kps):
                    for kp, desc in zip(kps, descriptors):
                        temp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, desc)
                        features.append(temp)
                query_feats[name] = features
            # if(len(query_feats)>10):
            #    break

        with open(query_name, "wb") as f:
            pickle.dump(query_feats, f)

    if method_name == 'sift' or method_name == 'root_sift':
        unserialized = {}
        for name in query_feats:
            unserialized[name] = [[], []]
            for f in query_feats[name]:
                kp = cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=f[1], _angle=f[2],
                                  _response=f[3], _octave=f[4], _class_id=f[5])
                desc = f[6]
                unserialized[name][0].append(kp)
                unserialized[name][1].append(desc)

            if len(unserialized[name][1]):
                unserialized[name][1] = np.vstack(unserialized[name][1])
        query_feats = unserialized

    return database_feats, query_feats


def main(database_dir, query_folder, ground_truth, method_name, forTest=False):
    if method_name == 'orb':
        method = Orb(max_features=2000)
        # min_features = 25
        th = 0.02
    elif method_name == 'surf':
        method = Surf()
        # min_features = 100
        th = 0.24
    elif method_name == 'sift':
        method = Sift()
        # min_features = 70
        th = 0.0341796875
        print("SIFT threshold might need to be tuned again")
    elif method_name == 'root_sift':
        method = RootSift()
        # min_features = 70
        th = 0.1  # 0.0341796875
    else:
        exit(1)

    data = Data(database_dir=database_dir, query_dir=query_folder)

    query_imgs = [[None, name] for im, name in data.query_imgs]
    database_imgs = [[None, name] for im, name in data.database_imgs]
    database_feats, query_feats = get_features(method, data, query_folder, database_dir, method_name)

    eval_array = []

    res = []
    for _, q_name in query_imgs:

        scores = method.retrieve_best_results(None, database_imgs, database_feats, query_feats[q_name])

        if method_name == 'sift' or method_name == 'root_sift':
            features_num = len(query_feats[q_name][0])
        else:
            features_num = len(query_feats[q_name])
        features_num = max(features_num, 1)

        if scores[0][1] / features_num < th:  # minimum number of features matched allowed (-1 otherwise)
            scores = [(-1, 0)]
        eval = evaluation(predicted=[s[0] for s in scores], actual=ground_truth[q_name])
        eval_array.append(eval)

        res.append([score for score, _ in scores])

        print(scores[:3], "   ", ground_truth[q_name], "   ", scores[0][1] / features_num)
        print(eval)
        #
        # import matplotlib.pyplot as plt
        # plt.subplot(1, 2, 1)
        # plt.imshow(plt.imread(query_folder + "/" + q_name + ".jpg"))
        # plt.subplot(1, 2, 2)
        # if len(scores) > 3:
        #     plt.imshow(plt.imread(database_dir + "/" + scores[0][0] + ".jpg"))
        # else:
        #     plt.imshow(np.ones((200, 200)).astype("uint8"))
        # plt.show()

    global_eval = np.mean(eval_array)
    print("----------------\nEvaluation: " + str(global_eval))

    q = [name for _, name in data.query_imgs]
    with open("result.pkl", "wb") as f:
        pickle.dump(res, f)
    with open("query.pkl", "wb") as f:
        pickle.dump(q, f)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', help='runs the code on the test dataset (only week3 by now)',
                        action='store_true')
    parser.add_argument('-use_orb', help='uses orb to match the images',
                        action='store_true')
    parser.add_argument('-use_surf', help='uses surf to match the images',
                        action='store_true')
    parser.add_argument('-use_sift', help='uses sift to match the images',
                        action='store_true')
    parser.add_argument('-use_root_sift', help='uses root sift to match the images',
                        action='store_true')
    args = parser.parse_args()

    # here we select the folder with the right database and queries according to the options
    if args.test:
        query_folder = "w5_test_random"
        ground_truth = ground_truth_W5
    else:
        query_folder = "w5_devel_random"
        ground_truth = ground_truth_W5

    database_dir = "w5_BBDD_random"

    print(query_folder)
    if args.use_orb:
        method_name = 'orb'
    elif args.use_surf:
        method_name = 'surf'
    elif args.use_sift:
        method_name = 'sift'
    elif args.use_root_sift:
        method_name = 'root_sift'
    else:
        raise ValueError("Unspecified matching method")

    if args.test:
        forTest = True
    else:
        forTest = False

    main(database_dir, query_folder, ground_truth, method_name, forTest)
