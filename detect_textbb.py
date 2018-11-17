import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_handler import Data
import ground_truth_text
from math import sqrt

def black_filter(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	sensitivity = 50
	lower_black = np.array([0,  0,  0])
	upper_black = np.array([255,255,sensitivity])
	mask = cv2.inRange(hsv, lower_black, upper_black)
	cv2.imshow('black',mask)

def white_filter(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	sensitivity = 50
	lower_white = np.array([0,  0,          255-sensitivity])
	upper_white = np.array([255,sensitivity,255])
	mask = cv2.inRange(hsv, lower_white, upper_white)
	cv2.imshow('white',mask)

def line_detection(im):

	lines_np = []
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	kernel_size = 11
	gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
	gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


	edges_im = cv2.Canny(gray,10,20) #50-100
	lines_im = edges_im.copy() * 0
	lines = cv2.HoughLinesP(edges_im,rho = 1,theta = 1*np.pi/180,threshold = 10,\
							minLineLength = 80,maxLineGap = 20)  #threshold = 100,minLength = 50,maxLineGap = 50

	if type(lines) == type(None):
		lines = []

	if len(lines):
		for i in range(lines.shape[0]):
			x1 = lines[i][0][0]
			y1 = lines[i][0][1]	
			x2 = lines[i][0][2]
			y2 = lines[i][0][3]
			if abs(y1-y2) < 5 or abs(x1-x2) < 5:
				line_np = [ np.array((x1,y1)), np.array((x2,y2)) ]
				lines_np.append(line_np)
				cv2.line(lines_im,(x1,y1),(x2,y2),255,4)

	cv2.imshow('gray',gray)
	cv2.imshow('edges_im',edges_im)
	cv2.imshow('lines',lines_im)
	#print(len(lines))
	return lines_np, lines_im, edges_im



def main():
	data = Data(database_dir= 'w5_BBDD_random', query_dir= 'w5_devel_random')
	gt = ground_truth_text.get_text_gt()

	# loop over database_imgs without overloading memory
	for im, im_name in data.database_imgs:
		#line_detection(im)
		white_filter(im)
		black_filter(im)
		cv2.imshow('input',im)

		cv2.waitKey()




if __name__ == "__main__":
	main()