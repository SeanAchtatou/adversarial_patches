import numpy as np
import math
import random
import imutils
import cv2



image = cv2.imread("initial_patches/square.png")
x, y, z = image.shape
imageover = np.random.randint(125,130,(x,y,z)).astype(np.uint8)
for i in range(0,360,2):
    im = image.copy()
    im = imutils.rotate_bound(im,i)
    x = im == [0,0,0]


    #cv2.imshow("P",imageover)
    #cv2.waitKey(0)
