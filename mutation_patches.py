import random
import numpy as np
import math
import cv2



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

def quart_hor(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    for i in range(0,x,int(x/2)):
        new_patch[i:i+int(x/4),:] = patch1[i:i+int(x/4),:]
        new_patch[i+int(x/4):i+int(x/2),:] = patch2[i+int(x/4):i+int(x/2),:]

    return new_patch

def quart_vert(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    for i in range(0,x,int(x/2)):
        new_patch[:,i:i+int(x/4)] = patch1[:,i:i+int(x/4)]
        new_patch[:,i+int(x/4):i+int(x/2)] = patch2[:,i+int(x/4):i+int(x/2)]

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


def lines_hor(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    for i in range(0,x,4):
        new_patch[i:i+1,:] = patch1[i:i+1,:]
        new_patch[i+2:i+3,:] = patch2[i+2:i+3,:]

    return new_patch

def lines_vert(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    for i in range(0,x,4):
        new_patch[:,i:i+1] = patch1[:,i:i+1]
        new_patch[:,i+2:i+3] = patch2[:,i+2:i+3]

    return new_patch

def eight_vert(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    for i in range(0,x,int(x/4)):
        new_patch[:,i:i+int(x/8)] = patch1[:,i:i+int(x/8)]
        new_patch[:,i+int(x/8):i+int(x/4)] = patch2[:,i+int(x/8):i+int(x/4)]

    return new_patch

def eight_hor(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    for i in range(0,x,int(x/4)):
        new_patch[i:i+int(x/8),:] = patch1[i:i+int(x/8),:]
        new_patch[i+int(x/8):i+int(x/4),:] = patch2[i+int(x/8):i+int(x/4),:]


    return new_patch

def channels_switch_1(patch1,patch2):
    x, y, _ = patch1.shape
    b,g,r = cv2.split(patch1)
    b_,g_,r_ = cv2.split(patch2)
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    new_patch[:,:,0] = g
    new_patch[:,:,1] = b_
    new_patch[:,:,2] = r
    return new_patch

def channels_switch_2(patch1,patch2):
    x, y, _ = patch1.shape
    b,g,r = cv2.split(patch1)
    b_,g_,r_ = cv2.split(patch2)
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    new_patch[:,:,0] = g_
    new_patch[:,:,1] = b
    new_patch[:,:,2] = r
    return new_patch


def mutate(patches):
    thr = math.floor(len(patches)/10)
    end = (len(patches) % thr) - 1
    patches_ = []

    for i in range(0,len(patches)):

        """if i == 49:
            x,y,_ = patches[i].shape
            x_,y_,_ = patches[-1].shape
            #v = int((size_patch-x)/2)
            #w = int((size_patch-x_)/2)

            #patches[i] = np.pad(patches[i],[(v,v),(v,v),(0,0)],"constant")
            #patches[-1] = np.pad(patches[-1],[(w,w),(w,w),(0,0)],"constant")
            patches[i] = cv2.resize(patches[i],(size_patch,size_patch))
            patches[-1] = cv2.resize(patches[-1],(size_patch,size_patch))
        else:
            x,y,_ = patches[i].shape
            x_,y_,_ = patches[i+1].shape
            #v = int((size_patch-x)/2)
            #w = int((size_patch-x_)/2)
            #patches[i] = np.pad(patches[i],[(v,v),(v,v),(0,0)],"constant")
            patches[i] = cv2.resize(patches[i],(size_patch,size_patch))
            #patches[i+1] = np.pad(patches[i+1],[(w,w),(w,w),(0,0)],"constant")
            patches[i+1] = cv2.resize(patches[i+1],(size_patch,size_patch))"""


        if i < thr:
            patches_.append(half_hor(patches[i],patches[i+1]))
        if thr <= i < thr*2:
            patches_.append(half_vert(patches[i],patches[i+1]))
        if thr*2 <= i < thr*3:
            patches_.append(lines_hor(patches[i],patches[i+1]))
        if thr*3 <= i < thr*4:
            patches_.append(lines_vert(patches[i],patches[i+1]))
        if thr*4 <= i < thr*5:
            patches_.append(many_hor(patches[i],patches[i+1]))
        if thr*5 <= i < thr*6:
            patches_.append(many_vert(patches[i],patches[i+1]))
        if thr*6 <= i < thr*7:
            patches_.append(quart_vert(patches[i],patches[i+1]))
        if thr*7 <= i < thr*8:
            patches_.append(quart_hor(patches[i],patches[i+1]))
        if thr*8 <= i < thr*9:
            patches_.append(eight_hor(patches[i],patches[i+1]))
        if thr*9 <= i < (thr*10)-1:
            patches_.append(eight_vert(patches[i],patches[i+1]))
        if i == ((thr*10)-1):
            patches_.append(eight_vert(patches[i],patches[-1]))


    return np.array(patches_)