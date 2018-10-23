import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


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


def retrieve_best_results(image_histH, dataset, K=10):
    scores = [(image, difference(image_histH, dataset[image]['histH'])) for image in dataset.keys()]
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


def test_ground_truth(ground_truth, museum_set, query_set):
    for id in ground_truth:
        gt_image = query_set[id]['image']
        museum_image = museum_set[ground_truth[id]]['image']
        rgb_gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        rgb_museum_image = cv2.cvtColor(museum_image, cv2.COLOR_BGR2RGB)
        plt.subplot(121), plt.imshow(rgb_gt_image)
        plt.subplot(122), plt.imshow(rgb_museum_image)
        plt.show()


def evaluation(predicted, actual, k=10):

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

    # test_ground_truth(ground_truth=ground_truth, museum_set=museum_set, query_set=query_set)
    eval_array = []
    for query in query_set:
        scores = retrieve_best_results(image_histH=query_set[query]['histH'], dataset=museum_set)

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