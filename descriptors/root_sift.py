import numpy as np

from .sift import Sift


class RootSift(Sift):
    def __init__(self):
        Sift.__init__(self)

    def detectAndCompute(self, img, eps=1e-7):
        keypoints, descriptors = self.sift_obj.detectAndCompute(img, None)

        if len(keypoints) == 0:
            return [], []

        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)

        return keypoints, descriptors

