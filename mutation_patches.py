import random
import numpy as np
import math
import cv2
import genetic_patch_old

from genetic_patch_old import thresholdB,thresholdG,thresholdR



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


def inside_outside_half(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = patch1.copy()
    size = int(x/2)
    new_patch[int(size/2):int(size/2)+size,int(size/2):int(size/2)+size] = patch2[int(size/2):int(size/2)+size,int(size/2):int(size/2)+size]
    return new_patch

def inside_outside_quart(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    size = int(x/8)
    for i in range(4):
        if i%2 == 0:
            new_patch[i*size:x-(i*size),i*size:x-(i*size)] = patch1[i*size:x-(i*size),i*size:x-(i*size)]
        else:
            new_patch[i*size:x-(i*size),i*size:x-(i*size)] = patch2[i*size:x-(i*size),i*size:x-(i*size)]
    return new_patch

def inside_outside_eight(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    size = int(x/16)
    for i in range(8):
        if i%2 == 0:
            new_patch[i*size:x-(i*size),i*size:x-(i*size)] = patch1[i*size:x-(i*size),i*size:x-(i*size)]
        else:
            new_patch[i*size:x-(i*size),i*size:x-(i*size)] = patch2[i*size:x-(i*size),i*size:x-(i*size)]
    return new_patch

def inside_outside_full(patch1,patch2):
    x, y, _ = patch1.shape
    new_patch = np.zeros((x,y,3)).astype(np.uint8)
    size = 1
    for i in range(x):
        if i%2 == 0:
            new_patch[i*size:x-(i*size),i*size:x-(i*size)] = patch1[i*size:x-(i*size),i*size:x-(i*size)]
        else:
            new_patch[i*size:x-(i*size),i*size:x-(i*size)] = patch2[i*size:x-(i*size),i*size:x-(i*size)]
    return new_patch


def muta(patch):
    color,r,g,b = genetic_patch_old.mean(patch)
    patch_size = int(patch.shape[0])
    maxx = max(g,b)
    R = np.random.randint(max(0,0),255,(patch_size,patch_size)).astype(np.uint8)
    G = np.random.randint(max(0,0),min(255,maxx+1),(patch_size,patch_size)).astype(np.uint8)
    B = np.random.randint(max(0,0),min(255,maxx+1),(patch_size,patch_size)).astype(np.uint8)
    patch = cv2.merge([B,G,R])
    return patch


def pixel_change(patch1,patch2):
    color,r,g,b = genetic_patch_old.mean(patch1)
    color_,r_,g_,b_ = genetic_patch_old.mean(patch2)
    r, r_ = min(r,r_), max(r_,r) + 1
    g, g_ = min(g,g_), max(g,g_) + 1
    b,b_ = min(b,b_), max(b,b_) + 1
    patch_size = int(patch1.shape[0])
    #R = np.random.randint(max(0,r),min(255,r_),(patch_size,patch_size)).astype(np.uint8)
    #G = np.random.randint(max(0,0),min(255,g_),(patch_size,patch_size)).astype(np.uint8)
    #B = np.random.randint(max(0,0),min(255,b_),(patch_size,patch_size)).astype(np.uint8)
    R = np.random.randint(0,255,(patch_size,patch_size)).astype(np.uint8)
    G = np.random.randint(0,255,(patch_size,patch_size)).astype(np.uint8)
    B = np.random.randint(0,255,(patch_size,patch_size)).astype(np.uint8)
    patch = cv2.merge([B,G,R])
    #patch = patch1  patch

    return patch

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
    x = [half_hor,half_vert,lines_hor,lines_vert,many_vert,many_hor,quart_vert,quart_hor,eight_hor,eight_vert,inside_outside_quart,inside_outside_eight,inside_outside_half,inside_outside_full]
    thr = math.floor(len(patches)/len(x))
    rest = ((len(patches)%11)+1)%2
    patches_ = []

    for i in range(0,len(patches)):

        """x = [half_hor,half_vert,lines_hor,lines_vert,many_vert,many_hor,quart_vert,quart_hor,eight_hor,eight_vert]
        for j in x:
            prob = np.random.rand()
            if prob < 0.5:
                patches_.append(j(patches[i],patches[i+1]))
        patches_.append(pixel_change(patches[i],patches[i+1]))"""
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

        """patches_.append(half_hor(patches[i],patches[i+1]))
        patches_.append(half_vert(patches[i],patches[i+1]))
        patches_.append(lines_hor(patches[i],patches[i+1]))
        patches_.append(lines_vert(patches[i],patches[i+1]))
        patches_.append(many_hor(patches[i],patches[i+1]))
        patches_.append(many_vert(patches[i],patches[i+1]))
        patches_.append(quart_vert(patches[i],patches[i+1]))
        patches_.append(quart_hor(patches[i],patches[i+1]))
        patches_.append(eight_hor(patches[i],patches[i+1]))
        patches_.append(eight_vert(patches[i],patches[i+1]))

        patches_.append(pixel_change(patches[i],patches[i+1]))"""


        n = np.random.rand()
        if n < 0.9:
            try:
                patches_.append(x[i%len(x)](patches[i],patches[i+1]))
            except:
                patches_.append(x[i%len(x)](patches[i],patches[-1]))
        else:
            patches_.append(patches[i])

        m = np.random.rand()
        if m < 0.2:
            try:
                patches_[i] = muta(patches_[i])
            except:
                pass


        """if i < thr:
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
            patches_.append(inside_outside_half(patches[i],patches[i+1]))
        if thr*9 <= i < thr*10:
            patches_.append(inside_outside_quart(patches[i],patches[i+1]))
        if thr*10 <= i < thr*11:
            patches_.append(inside_outside_eight(patches[i],patches[i+1]))
        if thr*11 <= i < thr*12:
            patches_.append(inside_outside_full(patches[i],patches[i+1]))
        if thr*12 <= i < thr*13:
            patches_.append(eight_hor(patches[i],patches[i+1]))
        if thr*13 <= i < (thr*14)+1:
            patches_.append(eight_vert(patches[i],patches[i+1]))
        if i == ((thr*14)+1):
            patches_.append(eight_vert(patches[i],patches[-1]))

        if np.random.rand() > 0.8:
            try:
                patches_[i] = muta(patches_[i])
            except:
                pass"""


    return np.array(patches_)