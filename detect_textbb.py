import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_handler import Data
import ground_truth_text
from math import sqrt

def intersection_over_union(bbox1, bbox2):
    #each bounding box should be given as (x1, y1, x2, y2). (x1, y1) the top left point, (x2, y2) the bottom right
    #intersection bbox
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])


    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    area1 = (bbox1[2]-bbox1[0] + 1) * (bbox1[3]-bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    union_area = area1 + area2 - intersection_area


    return intersection_area/union_area

def combine_boxes(box1,box2):
	#box = [x,y,w,h]
	x = min(box1[0],box2[0])
	y = min(box1[1],box2[1])
	w = max(box1[0]+box1[2],box2[0]+box2[2]) - x
	h = max(box1[1]+box1[3],box2[1]+box2[3]) - y

	return [x,y,w,h]


def black_filter(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	sensitivity = 50
	lower_black = np.array([0,  0,  0])
	upper_black = np.array([255,255,sensitivity])
	mask = cv2.inRange(hsv, lower_black, upper_black)
	cv2.imshow('black',mask)
	
	return mask

def white_filter(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	sensitivity = 50
	lower_white = np.array([0,  0,          255-sensitivity])
	upper_white = np.array([255,sensitivity,255])
	mask = cv2.inRange(hsv, lower_white, upper_white)
	cv2.imshow('white',mask)

	return mask


def line_detection(im):

	lines_np = []
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	kernel_size = 3
	gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
	gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
	gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
	gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
	gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


	edges_im = cv2.Canny(gray,10,50) #50-100
	lines_im = edges_im.copy() * 0
	lines = cv2.HoughLinesP(edges_im,rho = 1,theta = 1*np.pi/180,threshold = 20,\
							minLineLength = 50,maxLineGap = 10)  #threshold = 100,minLength = 50,maxLineGap = 50

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


def detect_text(im_rgb, color_msk):
	text_box    = [0,0,0,0]
	score       = -1000000
	candidates  = []
	horizontal_th = 5

	test_im = color_msk.copy()
	test_im = cv2.cvtColor(test_im, cv2.COLOR_GRAY2BGR)

	im2, contours, hierarchy = cv2.findContours(color_msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if w>20 and w < 300 and h>10 and h < 100:
			candidates.append([x,y,w,h])
			cv2.rectangle(test_im,(x,y),(x+w,y+h),(0,0,255),2)
			print(y+h)

	if not len(candidates):
		return text_box, score
	candidates_score = [0]*len(candidates)
	for i, cand in enumerate(candidates):
		for cand_ in candidates:
			x,y,w,h     = cand
			x_,y_,w_,h_ = cand_
			if(abs((y+h) - (y_+h_)) < horizontal_th):
				candidates_score[i] += 1

	max_score     = max(candidates_score)
	best_cand_idx = candidates_score.index( max_score )
	best_cand     = candidates[ best_cand_idx ]
	x,y,w,h       = best_cand
	best_union    = best_cand

	for i, cand in enumerate(candidates):
		x_,y_,w_,h_ = cand
		if(abs((y+h) - (y_+h_)) < horizontal_th):
			best_union = combine_boxes(best_union, cand)
	
	x,y,w,h       = best_union

	#cv2.drawContours(test_im, contours, -1, (0,255,0), 3)
	cv2.imshow('contours', test_im)

	center_score = -(abs(x+w/2-500)/500)*10  #==> -10 to 0
	
	score    = max_score + center_score
	text_box = best_union

	print('[',center_score,score,']')

	return text_box, score





def main():
	data = Data(database_dir= 'w5_BBDD_random', query_dir= 'w5_devel_random')
	gt = ground_truth_text.get_text_gt()
	iou_list = []
	# loop over database_imgs without overloading memory
	for im, im_name in data.database_imgs:
		x1gt, y1gt, x2gt, y2gt = gt[im_name]
		h,w = im.shape[0:2]
		ratio = 1000/w
		im  = cv2.resize(im,(1000,int(ratio*h)))#(ratio*w,ratio*h))
		#cv2.rectangle(im,(x1gt,y1gt),(x2gt,y2gt),(0,255,0),2)
		if 1:
			line_detection(im)
			w_msk = white_filter(im)
			b_msk = black_filter(im)
			
			w_box, w_score = detect_text(im, w_msk)
			b_box, b_score = detect_text(im, b_msk)

			if w_score>b_score:
				text_box = w_box
			else:
				text_box = b_box
			x,y,w,h = text_box
			#---------------------------------
			y = int(y-0.2*h)
			h = int(1.7*h)
			
			x = int(x - 0.2*h)
			w = int(w + 0.4*h)
			text_box = [x,y,w,h]
			#---------------------------------

			cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)

			rescaled_text_box = [int(i/ratio) for i in text_box]
			x,y,w,h = rescaled_text_box
			rescaled_text_box = [x,y,x+w,y+h]

			print(rescaled_text_box)
			print(gt[im_name])
			iou = intersection_over_union(rescaled_text_box, gt[im_name])
			print("iou: "+str(iou))
			iou_list.append(iou)
			
			cv2.imshow('input',im)
			print()
			cv2.waitKey()

	print(iou_list)
	print (np.mean(iou_list))

if __name__ == "__main__":
	main()