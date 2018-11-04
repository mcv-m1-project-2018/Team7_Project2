import cv2


class Sift:
    def __init__(self, matching_method='flann'):
        self.sift_obj = cv2.xfeatures2d.SIFT_create(nfeatures=100,
                                                    nOctaveLayers=3,
                                                    contrastThreshold=0.03,
                                                    edgeThreshold=10,
                                                    sigma=5)
        self.matching_method = matching_method
        #self.sift_obj.setMaxFeatures(100)


    def detectAndCompute(self, img):

        keypoints, descriptors = self.sift_obj.detectAndCompute(img, None)
        
        features_visualized = None
        features_visualized    = cv2.drawKeypoints(img,keypoints,features_visualized,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print(len(keypoints))
        cv2.imshow('sift_kps', features_visualized)
        cv2.waitKey(1)
        
        return keypoints, descriptors

    def match_features(self, descriptors1, descriptors2, threshold=0.9):
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
            index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
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
        scores     = []
        best_score = 0
        best_idx   = 0 
        for idx, [d_image, d_name] in enumerate( database_imgs ):
            d_desc = database_features[d_name][1]
            #print(q_feat[1][0].shape)
            #print(d_desc[0].shape)
            matches = self.match_features(q_feat[1], d_desc)
            scores.append((d_name, len(matches)))
            if(len(matches) > best_score):
                best_score = len(matches)
                best_idx   = idx
            
        scores.sort(key=lambda s: s[1], reverse=True)

        img3 = cv2.drawMatchesKnn(database_imgs[best_idx][0], 
                                  database_features[database_imgs[best_idx][1]][0],
                                  q_image,q_feat[0],
                                  matches,q_image)

        cv2.imshow('matches', img3)
        cv2.waitKey()
            #"""


        return scores[:K]

