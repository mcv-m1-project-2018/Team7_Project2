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
from descriptors.sift import Sift

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
            if method_name == 'sift':
                kps, descriptors = method.detectAndCompute(im)
                features = []
                if(len(kps)):
                    for kp,desc in zip(kps,descriptors):
                        temp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, desc)  
                        features.append(temp)
                database_feats[name] = features
            #if(len(database_feats)>2):
             #   break
                #database_feats[name] = method.detectAndCompute(im)
        with open(db_name, "wb") as f:
            pickle.dump(database_feats, f)

    if method_name == 'sift':
        unserialized = {}
        for name in database_feats:
            unserialized[name] = [[],[]]
            for f in database_feats[name]:
                kp   = cv2.KeyPoint(x=f[0][0],y=f[0][1],_size=f[1], _angle=f[2],
                                    _response=f[3], _octave=f[4], _class_id=f[5])
                desc = f[6]
                unserialized[name][0].append(kp)
                unserialized[name][1].append(desc)
            if(len( unserialized[name][1] )):
                unserialized[name][1] =  np.vstack( unserialized[name][1])
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
            if method_name == 'surf':
                query_feats[name] = method.detectAndCompute(im, adapt_threshold=True, step=1000, max_features=1000)
            if method_name == 'orb':
                query_feats[name] = method.detectAndCompute(im)
            if method_name == 'sift':
                kps, descriptors = method.detectAndCompute(im)
                features = []
                if(len(kps)):
                    for kp,desc in zip(kps,descriptors):
                        temp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, desc)  
                        features.append(temp)
                query_feats[name] = features
            #if(len(query_feats)>10):
            #    break

        with open(query_name, "wb") as f:
            pickle.dump(query_feats, f)

    if method_name == 'sift':
        unserialized = {}
        for name in query_feats:
            unserialized[name] = [[],[]]
            for f in query_feats[name]:
                kp   = cv2.KeyPoint(x=f[0][0],y=f[0][1],_size=f[1], _angle=f[2],
                                    _response=f[3], _octave=f[4], _class_id=f[5])
                desc = f[6]
                unserialized[name][0].append(kp)
                unserialized[name][1].append(desc)

            if len( unserialized[name][1] ):
                unserialized[name][1] = np.vstack( unserialized[name][1])
        query_feats = unserialized

    return database_feats, query_feats


def main(database_dir, query_folder, ground_truth, method_name, forTest = False):
    if method_name == 'orb':
        method = Orb()
        #min_features = 25
        th = 0.0205078125
    elif method_name == 'surf':
        method = Surf()
        #min_features = 100
        th = 0.1669921875
    elif method_name == 'sift':
        method = Sift()
        #min_features = 70
        th = 0.155


    data = Data(database_dir=database_dir, query_dir=query_folder)

    query_imgs    = [[im, name] for im, name in data.query_imgs]
    database_imgs = [[im, name] for im, name in data.database_imgs]
    database_feats, query_feats = get_features(method, data, query_folder, database_dir, method_name)

    results_hist = []

    eval_array = []

    res = []
    for q_im, q_name in query_imgs:
        scores = method.retrieve_best_results(q_im, database_imgs, database_feats, query_feats[q_name])
        results_hist.append(scores)
        if method_name == 'sift':
            features_num = len(query_feats[q_name][0])
        else:
            features_num = len(query_feats[q_name])
        features_num = max(features_num,1)
        if not args.week3:  # week 4 evaluation
            if scores[0][1] / features_num < th:  # minimum number of features matched allowed (-1 otherwise)
                scores = [(-1, 0)]
            eval = evaluation(predicted=[s[0] for s in scores], actual=ground_truth[q_name])
        else:  # week 3 evaluation
            eval = evaluation(predicted=[s[0] for s in scores], actual=[ground_truth[q_name]])
        eval_array.append(eval)

        res.append([score for score, _ in scores])

        if forTest:
            print(ground_truth[q_name])
            print(eval)

    global_eval = np.mean(eval_array)
    print("----------------\nEvaluation: " + str(global_eval))
    q = [name for _, name in data.query_imgs]
    with open("result.pkl", "wb") as f:
        pickle.dump(res, f)
    with open("query.pkl", "wb") as f:
        pickle.dump(q, f)

    if forTest:
        # ------------------------------------- tuning
        print('\n\n--- tuning ---')
        best_th = None
        best_score = 0
        for th_i in range(1,1024,1):
            eval_array = []
            th = th_i/1024
            for scores, [q_im, q_name] in zip(results_hist, query_imgs):
                if (method_name == 'sift'):
                    features_num = len(query_feats[q_name][0])
                else:
                    features_num = len(query_feats[q_name])  
                features_num = max(features_num,1)          
                if not args.week3:  # week 4 evaluation
                    if scores[0][1] / features_num < th:  # minimum number of features matched allowed (-1 otherwise)
                        scores = [(-1, 0)]
                    eval = evaluation(predicted=[s[0] for s in scores], actual=ground_truth[q_name])
                else:  # week 3 evaluation
                    eval = evaluation(predicted=[s[0] for s in scores], actual=[ground_truth[q_name]])
                eval_array.append(eval)

            global_eval = np.mean(eval_array)

            if global_eval > best_score:
                best_score = global_eval
                best_th = th
        print('best_th = ', best_th)
        print('best_score = ', best_score)
        # ------------------------------------- print details with best th
        eval_array = []
        for scores, [q_im, q_name] in zip(results_hist, query_imgs):
            if (method_name == 'sift'):
                features_num = len(query_feats[q_name][0])
            else:
                features_num = len(query_feats[q_name])      
            features_num = max(features_num,1)
            print('=====================')
            print('num_kps:',features_num)
            print('best match:', scores[0])
            if not args.week3:  # week 4 evaluation
                if scores[0][1] / features_num < best_th:  # minimum number of features matched allowed (-1 otherwise)
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
    parser.add_argument('-use_surf', help='uses ORB to match the images',
                        action='store_true')
    parser.add_argument('-use_sift', help='uses sift to match the images',
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
        query_folder = "query_test"
        ground_truth = ground_truth_W4          # dummy ground truth
    else:                                       # week 4 dev set
        query_folder = "query_devel_W4"
        ground_truth = ground_truth_W4
        
    print(query_folder)
    if args.use_orb:
        method_name = 'orb'
    elif args.use_surf:
        method_name = 'surf'
    elif args.use_sift:
        method_name = 'sift'

    if args.test:
        forTest = True
    else:
        forTest = False

    main(database_dir, query_folder, ground_truth, method_name, forTest)
