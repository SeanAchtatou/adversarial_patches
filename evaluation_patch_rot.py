import numpy as np
import math
import random
import imutils
import cv2



image = cv2.imread("initial_patches/square.png")

while True:
    print("ALl")
    for i in range(0,360,45):
        im = image.copy()
        im = imutils.rotate_bound(im,i)
        x, y, z = im.shape
        imageover = np.random.randint(125,130,(x,y,z)).astype(np.uint8)
        x = im == (0,0,0)
        im[x] = imageover[x]

        cv2.imshow("P",im)
        cv2.waitKey(0)

