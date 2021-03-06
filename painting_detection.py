from data_handler import Data
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from skimage.filters import sobel
import numpy as np
import math
import pickle

"""
Example code for the segmentation of the paintings.
"""


def rotateImage(image, angle, center):
    rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def get_painting_rotated(image, show=False, save_fig=False, imname=None):
    # rescale the image, works better and uses less memory
    shape_max = max(image.shape)
    if shape_max > 750:
        ratio = 750 / shape_max
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = gray[:, :, 2]

    # edge detection, binarization, some morphology and contour extraction
    edge = sobel(gray)
    edge = cv2.GaussianBlur(edge, (13, 13), 0)
    mean = edge.mean()
    edge_binary = 1.0 * (edge > mean * 1.4)
    kernel = np.ones((10, 10), np.uint8)
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
        angle = rect[2] + 90
    else:
        angle = rect[2]

    # Change the upper left edge of the bounding box and the rotation. The upper left edge changes based on tilting
    # of the image (again read the article). This is because the point 0 changes between the lower right corner
    # and the lower left corner based on the direction of the tilting of the image (execute the code to see the edges
    # of the image).
    # The definition of the height and the width change as the upper left edge also changes (they are calculated
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
    rotated_image = rotateImage(image, angle, upper_left_edge)
    cropped_image = rotated_image[upper_left_edge[1]:upper_left_edge[1] + h_rotated_bbox,
                                  upper_left_edge[0]:upper_left_edge[0] + w_rotated_bbox, :]

    angle2 = -angle + 180  # angle as defined in the slides
    if show:
        fig = plt.figure()

        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(image)
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

        cx = x + w/2
        cy = y + h/2
        ax1.plot([cx, cx + 200],
                 [cy, cy], color="red", linewidth=2)
        ax1.plot([cx, cx - 200],
                 [cy, cy], color="red", linewidth=2)
        ax1.plot([cx, cx + 200 * math.cos(math.radians(-angle2))],
                 [cy, cy + 200 * math.sin(math.radians(-angle2))], color="green", linewidth=2)
        ax1.plot([cx, cx + 200 * math.cos(math.radians(-angle2+180))],
                 [cy, cy + 200 * math.sin(math.radians(-angle2+180))], color="green", linewidth=2)
        ax1.text(0.04, 0.05, "angle2: " + "{0:.2f}".format(angle2), transform=ax1.transAxes, fontsize=13, color="red",
                 bbox=dict(facecolor="gray", alpha=0.5, edgecolor='none'))
        arc = pat.Arc((cx, cy), 200, 200, 0, theta1=-angle2, theta2=0, linewidth=2, color="red")
        ax1.add_patch(arc)

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
        ax1.imshow(cropped_image)

        if save_fig:
            fig.savefig("./test/" + imname + "composition.jpg")
            plt.close(fig)
        else:
            plt.show()

    return cropped_image, (upper_left_edge, h_rotated_bbox, w_rotated_bbox, angle,)


def main():
    database_dir = 'w5_BBDD_random'
    query_folder = "w5_test_random"

    data = Data(database_dir=database_dir, query_dir=query_folder)

    # frames = []

    for q_im, q_name in data.query_imgs:
        get_painting_rotated(q_im, show=True, imname=q_name, save_fig=False)
        # bbox = []
        # bbox.append(int(angle))
        # temp = []
        # for i in box:
        #     temp.append(tuple(i.astype("int32")))
        # bbox.append(temp)
        # frames.append(bbox)
        # print(bbox)

    # with open("./frames.pkl", "wb") as f:
    #     pickle.dump(frames, f)

    exit(0)


if __name__ == "__main__":
    main()


