import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from data_handler import Data
import ground_truth_text
from math import sqrt
from image_colorfulness import divide_measure_colorfulness

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
	lines = cv2.HoughLinesP(edges_im,rho = 1,theta = 1*np.pi/180,threshold = 10,\
							minLineLength = 30,maxLineGap = 2)  #threshold = 100,minLength = 50,maxLineGap = 50

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


def detect_text(im_rgb, color_msk, label):
	text_box    = [0,0,0,0]
	score       = -1000000
	candidates  = []
	
	horizontal_th = 3
	min_w = 20
	max_w = 300
	min_h = 10
	max_h = 100

	#msk = divide_measure_colorfulness(im_rgb)
	#msk = cv2.bitwise_not(msk)
	#color_msk = cv2.bitwise_and(color_msk, msk)
	#cv2.imshow('colorful',msk)

	#kernel    = np.ones((3,3),np.uint8)
	#color_msk = cv2.morphologyEx(color_msk, cv2.MORPH_CLOSE, kernel)
	#kernel_size = 3
	#color_msk = cv2.GaussianBlur(color_msk, (kernel_size, kernel_size), 0)


	test_im = color_msk.copy()
	test_im = cv2.cvtColor(test_im, cv2.COLOR_GRAY2BGR)

	im2, contours, hierarchy = cv2.findContours(color_msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		#cnt_perimeter = cv2.arcLength(cnt,True)
		#box_perimeter = 2*(w+h)
		if w>min_w and w < max_w and h>min_h and h < max_h:
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
			if(abs((y+h) - (y_+h_)) < horizontal_th) and \
			   intersection_over_union([x,y,x+w,y+h], [x_,y_,x_+w_,y_+h_]) < 0.2 and \
			   (h/h_ < 3 or h_/h < 3) :
			   #((w+h)/(w_*h_) < 2 or (w_+h_)/(w*h) < 2)

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

	cv2.drawContours(test_im, contours, -1, (0,255,0), 3)
	cv2.imshow('contours'+ label, test_im)

	image_midlle_x = im_rgb.shape[1]/2
	center_score = -(abs(x+w/2-image_midlle_x)/image_midlle_x)*10  #==> -10 to 0
	
	score    = max_score + center_score
	text_box = best_union

	print('[',center_score,score,']')

	return text_box, score


def posprocess_bbox(box, lines_np):
	new_box = box
	x,y,w,h = box
	x_,y_,w_,h_ = box

	INF = 1000000
	min_dis_up    = int(0.6*h)
	min_dis_dowm  = int(0.5*h)
	min_dis_Left  = int(0.07*w)
	min_dis_right = int(0.07*w)

	for line in lines_np:
		p1,p2 = line

		if abs(p1[1]-p2[1]) < 5: #horizontal line 
		# ---------- find closest up line minimize y-p1[1]
			if (y-p1[1]) >= 5 and (y-p1[1])<min_dis_up:
				min_dis_up = (y-p1[1])
		# ---------- find closest down line minimize p1[1] - (y+h)
			if (p1[1] - (y+h)) >= 5 and (p1[1] - (y+h))<min_dis_dowm:
				min_dis_dowm = (p1[1] - (y+h))

		if abs(p1[0]-p2[0]) < 5: #vertical line 
		# ---------- find closest left line minimize x-p1[0]
			if (x-p1[0]) >= 5 and (x-p1[0])<min_dis_Left:
				min_dis_Left = (x-p1[0])
		# ---------- find closest right line minimize p1[0] - (x+w)
			if (p1[0] - (x+w)) >= 5 and (p1[0] - (x+w))<min_dis_right:
				min_dis_right = (p1[0] - (x+w))

	min_dis_up = min(min_dis_up, y)

	y_ = y  - min_dis_up
	h_ = h_ + min_dis_dowm  + min_dis_up
	x_ = x  - min_dis_Left
	w_ = w_ + min_dis_right + min_dis_Left

	new_box = [x_,y_,w_,h_]

	return new_box



def main():
	data = Data(database_dir= 'w5_BBDD_random', query_dir= 'w5_devel_random')
	gt = ground_truth_text.get_text_gt()
	iou_list = []
	# loop over database_imgs without overloading memory
	for im, im_name in data.database_imgs:
		x1gt, y1gt, x2gt, y2gt = gt[im_name]
		h,w = im.shape[0:2]
		fixed_width = 1000
		ratio = fixed_width/w
		im  = cv2.resize(im,(fixed_width,int(ratio*h)))#(ratio*w,ratio*h))
		#cv2.rectangle(im,(x1gt,y1gt),(x2gt,y2gt),(0,255,0),2)
		if 1:#im_name =='ima_000036':
			lines_np, lines_im, edges_im = line_detection(im)
			w_msk = white_filter(im)
			b_msk = black_filter(im)
			
			w_box, w_score = detect_text(im, w_msk, '_white')
			b_box, b_score = detect_text(im, b_msk, '_black')

			if w_score>b_score:
				text_box = w_box
				print('white')
			else:
				text_box = b_box
				print('black')
			x,y,w,h = text_box
			#---------------------------------
			"""
			y = int(y-0.2*h)
			h = int(h+0.6*h)
			
			x = int(x - 0.04*w)
			w = int(w + 0.08*w)
			text_box = [x,y,w,h]
			"""
			text_box = posprocess_bbox(text_box, lines_np)
			x,y,w,h = text_box
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
			#print()
			#if iou<0.5:
			#cv2.waitKey()

	print(iou_list)
	print (np.mean(iou_list))

if __name__ == "__main__":
	main()