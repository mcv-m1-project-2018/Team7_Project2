import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


class Img_iterator():
    def __init__(self, imgs_dir = ''):
        self.im_idx     = 0
        self.imgs_dir   = imgs_dir
        self.imgs_files = os.listdir(imgs_dir)

    def __iter__(self):
        return self

    def __next__(self):
        im_idx = self.im_idx
        self.im_idx += 1

        if im_idx == len(self.imgs_files):
            self.im_idx = 0
            raise StopIteration
        else:
            file_name, ext = os.path.splitext(self.imgs_files[im_idx])
            im = cv2.imread(os.path.join(self.imgs_dir,file_name + ext))
            return im, file_name

    def read_img(self, img_file):        
        im = cv2.imread(os.path.join(self.imgs_dir,img_file))
        return im


class Data():
    def __init__(self, database_dir, query_dir):
        self.database_imgs = Img_iterator(database_dir)
        self.query_imgs    = Img_iterator(query_dir)



def main():
    data = Data(database_dir= 'museum_set_random', query_dir= 'query_devel_random')

    # open image with file_name
    cv2.imshow('ima_000000', data.database_imgs.read_img('ima_000000.jpg'))
    cv2.waitKey()

    # loop over database_imgs without overloading memory
    for im, im_name in data.database_imgs:
        cv2.imshow(im_name, im)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # same with data.query_imgs


if __name__ == "__main__":
    main()