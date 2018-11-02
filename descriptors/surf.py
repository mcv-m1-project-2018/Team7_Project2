import cv2


class Surf():
    """
    Class to compute surf features. Avoids creating cv2's surf object mode than once.
    """
    def __init__(self, hessian_threshold=300):
        self.surf_obj = cv2.xfeatures2d.SURF_create()
        self.surf_obj.setHessianThreshold(hessian_threshold)

    def set_hessian_threshold(self, thr):
        self.surf_obj.setHessianThreshold(thr)

    def get_hessian_threshold(self):
        return self.surf_obj.getHessianThreshold()

    def detectAndCompute(self, img, adapt_threshold=False, step=1000, max_features=1000):
        """
        Computes features, adapts the hessian threshold so the images have less than max_features features. If there
        are too many descriptors the KNN matcher takes too long. The keypoints are not used and it's not easy to
        pickle them, we just ignore them here.
        :param img: image
        :param adapt_threshold: adapts the threshold so there is always less than max_features features (for speed
        reasons)
        :param step: how fast the hessian threshold is adapted
        :param max_features: maximum number of features
        :return: descriptors.
        """
        keypoints, descriptors = self.surf_obj.detectAndCompute(img, None)

        if adapt_threshold:
            thr = 200
            old = self.get_hessian_threshold()
            while len(descriptors) > max_features:
                thr += step
                self.set_hessian_threshold(thr)
                keypoints, descriptors = self.surf_obj.detectAndCompute(img, None)
            self.set_hessian_threshold(old)
        return descriptors

    @staticmethod
    def match_features(descriptors1, descriptors2, threshold=0.75):
        """
        Matches features with KNN. We could try using FLANN
        TODO: try FLANN
        :param descriptors1:
        :param descriptors2:
        :param threshold: some threshold thing just leave it be
        :return: matches above the threshold
        """
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        if len(matches) == 0:
            return []

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

