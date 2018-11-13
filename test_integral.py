import cv2
import numpy as np

snap = np.ones((6,3),dtype='uint8')
print(snap)

integral_image = cv2.integral( snap )
print(integral_image)

# if you dont want the top row/left column pad
print(integral_image)

print("Sum of region (0,1)-(2,4)")
#Bottom right + top left - top right - bottom left
print(integral_image[5,3]+integral_image[1,0] - integral_image[1,3] - integral_image[5,0])