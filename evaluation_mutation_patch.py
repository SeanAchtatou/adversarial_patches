import random
import math
import numpy as np
import cv2
import os

def mean(image):
    b,g,r = cv2.split(image)
    mean_r = int(np.mean(r))
    mean_g = int(np.mean(g))
    mean_b = int(np.mean(b))

    if (mean_r >= (mean_g and mean_b)):
        return "R",mean_r,mean_g,mean_b
    if (mean_g >= (mean_b and mean_r)):
        return "G",mean_r,mean_g,mean_b
    if (mean_b >= (mean_g and mean_r)):
        return "B",mean_r,mean_g,mean_b


images_dir = "images_"
images = [cv2.imread(os.path.join(images_dir,i)) for i in os.listdir(images_dir)]
images = images[1]
form = "square"

thresholdR, thresholdG, thresholdB = 100,100,100
color,r,g,b = mean(images)
if color == "R":
    thresholdR = 150
if color == "G":
    thresholdG = 150
if color == "B":
    thresholdB = 150

image_size = images.shape[0]

patches = []
for i in range(100):
    if form == "square":
        patch_size = int(image_size/2)
        R = np.random.randint(max(0,r-thresholdR),min(255,r+thresholdR),(patch_size,patch_size)).astype(np.uint8)
        G = np.random.randint(max(0,g-thresholdG),min(255,g+thresholdG),(patch_size,patch_size)).astype(np.uint8)
        B = np.random.randint(max(0,b-thresholdB),min(255,b+thresholdB),(patch_size,patch_size)).astype(np.uint8)
        patch = cv2.merge([B,G,R])
        patches.append(patch)


patches[36] = np.zeros((int(image_size/2),int(image_size/2),3)).astype(np.uint8)


def half_hor(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)

    new_patch[:int(x/2),:] = patch1[:int(x/2),:]
    new_patch[int(x/2):,:] = patch2[int(x/2):,:]


    return new_patch

def half_vert(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    new_patch[:,:int(x/2)] = patch1[:,:int(x/2)]
    new_patch[:,int(x/2):] = patch2[:,int(x/2):]


    return new_patch

def many_hor(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)

    new_patch[:int(x/2),:int(x/2)] = patch1[:int(x/2),:int(x/2)]

    new_patch[int(x/2):,:int(x/2)] = patch2[int(x/2):,:int(x/2)]

    new_patch[int(x/2):,int(x/2):] = patch1[int(x/2):,int(x/2):]

    new_patch[:int(x/2),int(x/2):] = patch2[:int(x/2),int(x/2):]

    return new_patch

def many_vert(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)

    new_patch[:int(x/2),:int(x/2)] = patch2[:int(x/2),:int(x/2)]

    new_patch[int(x/2):,:int(x/2)] = patch1[int(x/2):,:int(x/2)]

    new_patch[int(x/2):,int(x/2):] = patch2[int(x/2):,int(x/2):]

    new_patch[:int(x/2),int(x/2):] = patch1[:int(x/2),int(x/2):]

    return new_patch

def channels_switch(patch1,patch2):
    x, y, _ = patch1.shape
    b,g,r = cv2.split(patch1)
    b_,g_,r_ = cv2.split(patch2)
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    new_patch[:,:,0] = b
    new_patch[:,:,1] = g_
    new_patch[:,:,2] = r_



    return new_patch

def mutate(patches):
    patches_ = []
    for i in range(0,len(patches)-1):
        if i < 10:
            patches_.append(half_hor(patches[i],patches[i+1]))
        if 10 <= i < 20:
            patches_.append(half_vert(patches[i],patches[i+1]))
        if 20 <= i < 30:
            patches_.append(many_hor(patches[i],patches[i+1]))
        if 30 <= i < 40:
            patches_.append(many_vert(patches[i],patches[i+1]))
        if 40 <= i < 50:
            patches_.append(channels_switch(patches[i],patches[i+1]))

    patches_ = np.append(patches_,patches_,axis=0)
    np.random.shuffle(patches_)


    return patches_




