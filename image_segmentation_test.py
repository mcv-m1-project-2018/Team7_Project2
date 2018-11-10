from data_handler import Data
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from skimage.filters import sobel
import numpy as np
import math

"""
Example code for the segmentation of the paintings.
"""


def rotateImage(image, angle, center):
    rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


database_dir = 'w5_BBDD_random'
query_folder = "w5_devel_random"

data = Data(database_dir=database_dir, query_dir=query_folder)


for q_im, q_name in data.query_imgs:

    # rescale the image, works better and uses less memory
    shape_max = max(q_im.shape)
    if shape_max > 1000:
        ratio = 750 / shape_max
        q_im = cv2.resize(q_im, (0, 0), fx=ratio, fy=ratio)

    gray = cv2.cvtColor(q_im, cv2.COLOR_BGR2GRAY)

    # edge detection, binarization, some morphology and contour extraction
    edge = sobel(gray)
    mean = edge.mean()
    edge_binary = 1.0 * (edge > mean*4)
    kernel = np.ones((10, 10), np.uint8)
    edge_binary = cv2.morphologyEx(np.uint8(edge_binary), cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    edge_binary = cv2.morphologyEx(np.uint8(edge_binary), cv2.MORPH_CLOSE, kernel)

    _, contours, _ = cv2.findContours(np.uint8(edge_binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # find the biggest contour and fill it.
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_contour = contour
            max_area = area
    cv2.fillPoly(edge_binary, pts=[max_contour], color=1)

    # extract the bbox of the biggest contour, and also adjust the angle
    x, y, w, h = cv2.boundingRect(max_contour)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)

    # Determine the angle of the bounding box. The angle that returns minAreaRect is always between 0 and -90, so we
    # can't know if it's tilted to the right or to the left. Don't worry, we can determine the angle of the rotation
    # by its value. Values close to 0 (more than -45 degrees) mean that the painting is tilted to the left, as the angle
    # between the lower side of the bbox and the horizontal reference line is smaller. Values closer to -90 (less than
    # -45) mean that the painting is tilted to the right, as it is now calculated as the angle between the right side of
    # the painting and the horizontal reference line. We COULD determine the angle of the rotation wrong if the
    # painting was tilted more than 45 degrees, but then it would be impossible to determine the right rotation as we
    # can't know when a painting is standing or sideways.

    # Take a look to this article, it helps understanding the rotation of the bboxes that minAreaRect returns:
    # https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/

    if rect[2] < -45:
        angle = rect[2]+90
    else:
        angle = rect[2]

    # Change the upper left edge of the bounding box and the rotation. The upper left edge changes based on tilting
    # of the image (again read the article). This is because the point 0 changes between the lower right corner
    # and the lower left corner based on the direction of the tilting of the image (execute the code to see the edges
    # of the image).
    # The definition of the height and the width changes as the upper left edge also changes (they are calculated
    # using the euclidean distance).
    # This is REALLY confusing. Please read the article.
    if angle <= 0:
        upper_left_edge = box[1].astype('int32')
        w_rotated_bbox = int(np.linalg.norm(box[2] - box[1]))
        h_rotated_bbox = int(np.linalg.norm(box[2] - box[3]))
    else:
        upper_left_edge = box[2].astype('int32')
        w_rotated_bbox = int(np.linalg.norm(box[2] - box[3]))
        h_rotated_bbox = int(np.linalg.norm(box[2] - box[1]))

    # the upper left edge (of the bbox) is the center of rotation
    rotated_image = rotateImage(q_im, angle, upper_left_edge)

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(q_im)
    ax1.scatter(box[0][0], box[0][1])
    ax1.scatter(box[1][0], box[1][1])
    ax1.scatter(box[2][0], box[2][1])
    ax1.scatter(box[3][0], box[3][1])
    ax1.text(0.04, 0.90, "angle: " + "{0:.2f}".format(angle), transform=ax1.transAxes, fontsize=13, color="magenta",
             bbox=dict(facecolor="gray", alpha=0.5, edgecolor='none'))
    ax1.text(0.04, 0.80, "minAreaRect angle: " + "{0:.2f}".format(rect[2]), transform=ax1.transAxes, fontsize=13,
             color="lime", bbox=dict(facecolor="gray", alpha=0.5, edgecolor='none'))
    ax1.add_patch(pat.Rectangle(upper_left_edge, w_rotated_bbox, h_rotated_bbox,  # rotated bbox
                                linewidth=2, edgecolor='r', facecolor='none', angle=angle))
    ax1.add_patch(pat.Rectangle((x, y), w, h,
                                linewidth=2, edgecolor='b', facecolor='none'))  # bbox without rotation
    ax1.plot([box[0, 0], box[0, 0]+90],  # minAreaRect angle
             [box[0, 1], box[0, 1]], color="lime", linewidth=2)
    ax1.plot([box[0, 0], box[0, 0] + 90 * math.cos(math.radians(rect[2]))],
             [box[0, 1], box[0, 1] + 90 * math.sin(math.radians(rect[2]))], color="lime", linewidth=2)
    ax1.plot([upper_left_edge[0], upper_left_edge[0] + 90],  # angle
             [upper_left_edge[1], upper_left_edge[1]], color="magenta", linewidth=2)
    ax1.plot([upper_left_edge[0], upper_left_edge[0] + 90 * math.cos(math.radians(angle))],
             [upper_left_edge[1], upper_left_edge[1] + 90 * math.sin(math.radians(angle))], color="magenta", linewidth=2)
    # these are the corners of the bbox returned by minAreaRect. Notice how the points of the image change based on
    # the tilting of the image. This is because opencv hates us and wants to make us scream in agony.
    ax1.annotate("0", (box[0][0] - 10, box[0][1] - 10), fontsize=13)
    ax1.annotate("1", (box[1][0] - 10, box[1][1] - 10), fontsize=13)
    ax1.annotate("2", (box[2][0] - 10, box[2][1] - 10), fontsize=13)
    ax1.annotate("3", (box[3][0] - 10, box[3][1] - 10), fontsize=13)

    ax1 = fig.add_subplot(2, 3, 2)
    ax1.imshow(edge, cmap='gray')

    ax1 = fig.add_subplot(2, 3, 3)
    ax1.imshow(edge_binary, cmap='gray')

    ax1 = fig.add_subplot(2, 3, 4)
    ax1.imshow(rotated_image)
    ax1.add_patch(pat.Rectangle(upper_left_edge, w_rotated_bbox, h_rotated_bbox,
                                linewidth=2, edgecolor='r', facecolor='none'))

    ax1 = fig.add_subplot(2, 3, 5)
    box = box.astype('int32')

    copped_image = rotated_image[upper_left_edge[1]:upper_left_edge[1] + h_rotated_bbox,
                                 upper_left_edge[0]:upper_left_edge[0] + w_rotated_bbox, :]

    ax1.imshow(copped_image)

    plt.show()

exit(0)

