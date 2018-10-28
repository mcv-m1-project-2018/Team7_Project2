import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import feat_wavelet_hash


def read_dataset(path, visualize=False):
    files = os.listdir(path)
    dataset = {}
    for file in files:
        image = cv2.imread(os.path.join(path,file))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist(images=[hsv_image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        histS = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        histV = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        sum_hist = histH.sum()
        dataset[file.strip('.jpg')] = {'image': image,
                                        'histH': histH / sum_hist,
                                        'histS': histS / sum_hist,
                                        'histV': histV / sum_hist}

        if (visualize):
            RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.subplot(221), plt.imshow(RGB_image)
            plt.subplot(222), plt.plot(histH / sum_hist, )
            plt.subplot(223), plt.plot(histS / sum_hist)
            plt.subplot(224), plt.plot(histV / sum_hist)
            plt.show()
    return dataset


def difference(image1_histH, image2_histH, metric=cv2.HISTCMP_CORREL):
    return cv2.compareHist(image1_histH, image2_histH, metric)


def retrieve_best_results(query_image, dataset, K=10):
    query_hash = feat_wavelet_hash.get_hash(query_image)
    scores = []
    for id in dataset.keys():
        dataset_image = dataset[id]['image']
        dataset_hash = feat_wavelet_hash.get_hash(dataset_image)
        score = (id, feat_wavelet_hash.compare_wavelet_hashing(q_hash=query_hash, im_hash=dataset_hash))
        scores.append(score)
    scores.sort(key=lambda s: s[1], reverse=False)
    return scores[:K]


def show_results(scores, museum_set, query_image, ground_truth_image):

    RGB_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    plt.subplot(341), plt.axis('off'), plt.imshow(RGB_image)

    bool_array = feat_wavelet_hash.get_hash(query_image).hash.flatten()
    int_array = []
    for i in bool_array:
        if i: int_array.append(1)
        else: int_array.append(0)
    plt.subplot(342), plt.axis('off')

    t = np.arange(0, len(int_array))
    x = int_array
    plt.plot(t, x, '-', lw=1, color='b')

    for i in range(5):
        RGB_image = cv2.cvtColor(museum_set[scores[i][0]]['image'], cv2.COLOR_BGR2RGB)
        plt.subplot(3,4,3+2*i), plt.axis('off'), plt.imshow(RGB_image)

        bool_array = feat_wavelet_hash.get_hash(museum_set[scores[i][0]]['image']).hash.flatten()
        int_array2 = []
        for j in bool_array:
            if j:
                int_array2.append(1)
            else:
                int_array2.append(0)
        plt.subplot(3,4,4+2*i), plt.axis('off')
        x2 = int_array2
        plt.plot(t, x2, '-', lw=1, color='r')
        plt.plot(t, x, '-', lw=1, color='b')




    plt.show()


def test_ground_truth(ground_truth, museum_set, query_set):
    for id in ground_truth:
        gt_image = query_set[id]['image']
        height_gt = np.size(gt_image, 0)
        width_gt = np.size(gt_image, 1)
        print("------------------------------------------------------------")
        print("Query Image dimensions:  width-> "+str(width_gt)+",   height-> "+str(height_gt)+",   aspect ratio-> "+str(width_gt/height_gt))
        museum_image = museum_set[ground_truth[id]]['image']
        height = np.size(museum_image, 0)
        width = np.size(museum_image, 1)
        print("Museum Image dimensions:  width-> " + str(width) + ",   height-> " + str(height) + ",   aspect ratio-> " + str(width / height))
        rgb_gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        rgb_museum_image = cv2.cvtColor(museum_image, cv2.COLOR_BGR2RGB)
        plt.subplot(121), plt.imshow(rgb_gt_image)
        plt.subplot(122), plt.imshow(rgb_museum_image)
        plt.show()


def evaluation(predicted, actual, k=10):

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def main():

    museum_set = read_dataset('museum_set_random')
    query_set = read_dataset('query_devel_random')

    ground_truth ={ #Annotations: Image in the query set -> Correct image in the museum set
        'ima_000000': 'ima_000076',
        'ima_000001': 'ima_000105',
        'ima_000002': 'ima_000034',
        'ima_000003': 'ima_000083',
        'ima_000004': 'ima_000109',
        'ima_000005': 'ima_000101',
        'ima_000006': 'ima_000057',
        'ima_000007': 'ima_000027',
        'ima_000008': 'ima_000050',
        'ima_000009': 'ima_000084',
        'ima_000010': 'ima_000025',
        'ima_000011': 'ima_000060',
        'ima_000012': 'ima_000045',
        'ima_000013': 'ima_000099',
        'ima_000014': 'ima_000107',
        'ima_000015': 'ima_000044',
        'ima_000016': 'ima_000065',
        'ima_000017': 'ima_000063',
        'ima_000018': 'ima_000111',
        'ima_000019': 'ima_000092',
        'ima_000020': 'ima_000012',
        'ima_000021': 'ima_000022',
        'ima_000022': 'ima_000087',
        'ima_000023': 'ima_000085',
        'ima_000024': 'ima_000013',
        'ima_000025': 'ima_000039',
        'ima_000026': 'ima_000103',
        'ima_000027': 'ima_000006',
        'ima_000028': 'ima_000062',
        'ima_000029': 'ima_000041',
    }

    #test_ground_truth(ground_truth=ground_truth, museum_set=museum_set, query_set=query_set)
    eval_array = []
    for query in query_set:
        scores = retrieve_best_results(query_image=query_set[query]['image'], dataset=museum_set)

        eval = evaluation(predicted=[s[0] for s in scores], actual=[ground_truth[query]])
        print(eval)
        eval_array.append(eval)
        show_results(scores=scores,
                     museum_set=museum_set,
                     query_image=query_set[query]['image'],
                     ground_truth_image=museum_set[ground_truth[query]]['image'])

    global_eval = np.mean(eval_array)
    print("----------------\nEvaluation: "+str(global_eval))



if __name__ == "__main__":
    main()