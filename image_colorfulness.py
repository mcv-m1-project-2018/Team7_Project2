import cv2
import numpy as np
from descriptors.feat_hsv_hst import get_nonoverllaped_regions
import os

import matplotlib.pyplot as plt

from data_handler import Data

import ground_truth_text


def image_colorfulness(image):
	(B, G, R) = cv2.split(image.astype("float"))
	rg = np.absolute(R - G)
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	# combine the mean and standard deviations
	stdRoot  = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot #+ (0.03 * meanRoot)


def divide_measure_colorfulness(im):
	slices = get_nonoverllaped_regions(im.shape,100)
	msk = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	for slice in slices:
		x,y,w,h = slice
		crop = np.copy(im[x:x+w,y:y+h])
		c = image_colorfulness(crop) * 10
		print(c)
		for i in range(x,x+w):
			for j in range(y,y+h):
				msk[i,j] = c

	cv2.imshow('msk',msk)


def main():
	data = Data(database_dir= 'w5_BBDD_random', query_dir= 'w5_devel_random')
	gt = ground_truth_text.get_text_gt()

	# loop over database_imgs without overloading memory
	#for im, im_name in data.database_imgs:
	for im, im_name in data.query_imgs:
		if im_name == 'ima_000023':
			x1gt, y1gt, x2gt, y2gt = gt[im_name]
			cv2.imshow('im',im)
			divide_measure_colorfulness(im)
			print(image_colorfulness(im))
			cv2.waitKey()


if __name__ == "__main__":
	main()