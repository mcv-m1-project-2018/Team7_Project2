import cv2


class Orb:
    """
    Class to compute orb features. Avoids creating cv2's ORB object more than once.
    """
    def __init__(self, max_features=1000, matching_method='knn'):
        """
        :param matching_method: possible values : knn, bf, flann
        """
        self.orb_obj = cv2.ORB_create()
        self.matching_method = matching_method
        self.orb_obj.setMaxFeatures(max_features)

    def detectAndCompute(self, img):
        """
        Computes the features. The keypoints are not used and it's not easy to pickle them, we just ignore them here.
        :param img: image
        :return: descriptors.
        """
        keypoints, descriptors = self.orb_obj.detectAndCompute(img, None)

        return descriptors

    def match_features(self, descriptors1, descriptors2, threshold=0.75):
        """
        Matches features with several methods. Brute force KNN seems to be the fastest one.
        :param descriptors1:
        :param descriptors2:
        :param threshold: some threshold thing just leave it be
        :return: matches above the threshold
        """
        matches_good = []

        if self.matching_method == 'knn':
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            if len(matches) == 0:
                return []

            matches_good = []
            for m, n in matches:
                if m.distance < threshold * n.distance:
                    matches_good.append([m])

        if self.matching_method == 'bf':
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            matches_good = bf.match(descriptors1, descriptors2)

        if self.matching_method == 'flann':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)

            matches_good = []
            for m, n in matches:
                if m.distance < threshold * n.distance:
                    matches_good.append([m])

        return matches_good

    def retrieve_best_results(self, q_image, database_imgs, database_features, q_feat, K=10):
        """
        Retrieves the top K matches between the image and the database.
        :param q_image: query image (not used should be deleted)
        :param database_imgs:  database images
        :param database_features: dictionary image_name-features
        :param q_feat: query image features
        :param K: retrieve the top K results
        :return: top K results
        """
        scores = []

        for d_image, d_name in database_imgs:
            d_desc = database_features[d_name]
            matches = self.match_features(q_feat, d_desc)
            scores.append((d_name, len(matches)))

        scores.sort(key=lambda s: s[1], reverse=True)

        return scores[:K]







