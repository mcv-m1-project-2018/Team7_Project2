import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_handler import Data
import ground_truth_text
from math import sqrt


def line_detection(im):

	lines_np = []
	edges_im = cv2.Canny(im,20,100) #50-100
	lines_im = edges_im.copy() * 0
	lines = cv2.HoughLinesP(edges_im,rho = 1,theta = 1*np.pi/180,threshold = 100,\
							minLineLength = 30,maxLineGap = 0)  #threshold = 100,minLength = 50,maxLineGap = 50

	if type(lines) == type(None):
		lines = []

	if len(lines):
		for i in range(lines.shape[0]):
			x1 = lines[i][0][0]
			y1 = lines[i][0][1]	
			x2 = lines[i][0][2]
			y2 = lines[i][0][3]
			line_np = [ np.array((x1,y1)), np.array((x2,y2)) ]
			lines_np.append(line_np)
			cv2.line(lines_im,(x1,y1),(x2,y2),255,4)

	cv2.imshow('lines',lines_im)
	#print(len(lines))
	return lines_np, lines_im, edges_im



def main():
	data = Data(database_dir= 'w5_BBDD_random', query_dir= 'w5_devel_random')
	gt = ground_truth_text.get_text_gt()

	# loop over database_imgs without overloading memory
	for im, im_name in data.database_imgs:
		line_detection(im)
		cv2.imshow('input',im)

		cv2.waitKey()




if __name__ == "__main__":
	main()