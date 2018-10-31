#See https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html for details

import numpy as np
import cv2

def find_keypoints_harris(image, subpixel_accuracy=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    harris_image = cv2.cornerHarris(src=gray,
                           blockSize= 2, #size of the neighbourhood for corner detection
                           ksize=3, #Aperture parameter of Sobel derivative used
                           k=0.04) #Harris detector free parameter in the equation R=det(M)−k(trace(M))2 TODO: Tune

    # Threshold for an optimal value, it may vary depending on the image. TODO: Tune
    retval, corners_image = cv2.threshold(harris_image, 0.01 * harris_image.max(), 255, 0)

    corners_image = np.uint8(corners_image)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corners_image)

    #dst = cv2.dilate(dst,None)

    if(subpixel_accuracy):

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

        corners = np.int0(corners)

    else:
        corners = np.int0(centroids)

    corners = [c[::-1] for c in corners] #invert (y, x) -> (x, y)
    return corners



if __name__ == "__main__":
    image = cv2.imread('../museum_set_random/ima_000041.jpg')
    corners = find_keypoints_harris(image, subpixel_accuracy=False)
    for point in corners:
        print(point)
        image[point[0], point[1]] = (0, 0, 255)
    corners = find_keypoints_harris(image)
    for point in corners:
        print(point)
        image[point[0], point[1]] = (0, 255, 0)
    cv2.imshow("corners", image)
    cv2.waitKey(0)

