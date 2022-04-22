import numpy as np
import math

import random
import os

import imutils
import cv2
import csv
import numpy as np
import math
import mutation_patches
import tensorflow as tf

from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow import keras


random.seed(1756)
patch_dir = "patches"
p_form = ["p_circle","p_rect","p_tri","p_sqr"]
images_dir = "images_"

def folder_creation():
    path = os.listdir()
    try:
        if patch_dir not in path:
            print("\033[93m     Folder for the \033[4mpatches\033[0m \033[93m have been created.\033[0m")
            os.mkdir(patch_dir)
            for i in p_form:
                os.mkdir(f"{patch_dir}/{i}")
        else:
            print("\033[93m     Folder for \033[4mpatches\033[0m \033[93m already exists.  \033[0m")

    except:
        print("\033[91m An error occured. \033[0m")


thresholdR, thresholdG, thresholdB = 100,100,100
def initial_patch(image,form):
    global thresholdR, thresholdG, thresholdB
    thresholdR, thresholdG, thresholdB = 100,100,100
    color,r,g,b = mean(image)
    if color == "R":
        thresholdR = 150
    if color == "G":
        thresholdG = 150
    if color == "B":
        thresholdB = 150

    image_size = image.shape[0]

    if form == "square":
        patch_size = int(image_size/2)
        R = np.random.randint(max(0,r-thresholdR),min(255,r+thresholdR),(patch_size,patch_size)).astype(np.uint8)
        G = np.random.randint(max(0,g-thresholdG),min(255,g+thresholdG),(patch_size,patch_size)).astype(np.uint8)
        B = np.random.randint(max(0,b-thresholdB),min(255,b+thresholdB),(patch_size,patch_size)).astype(np.uint8)
        patch = cv2.merge([B,G,R])

        return patch

    if form == "circle":
        patch_size = int(image_size/3)
        radius = patch_size
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x**2 + y**2 > radius**2
        cx, cy = radius, radius
        patch_size = len(index[0])

        R = np.random.randint(max(0,r-thresholdR),min(255,r+thresholdR),(patch_size,patch_size)).astype(np.uint8)
        G = np.random.randint(max(0,g-thresholdG),min(255,g+thresholdG),(patch_size,patch_size)).astype(np.uint8)
        B = np.random.randint(max(0,b-thresholdB),min(255,b+thresholdB),(patch_size,patch_size)).astype(np.uint8)
        R[:,:][index] = 0
        G[:,:][index] = 0
        B[:,:][index] = 0

        patch = cv2.merge([B,G,R])

        return patch


def calculate_diff(a,b):
    x = 0
    count = 0
    for i in a:
        x += (i-b[count])**2
        count += 1

    #cc = tf.keras.metrics.CategoricalCrossentropy()
    #ch = tf.keras.losses.CategoricalHinge()
    #hh = tf.keras.losses.Hinge()
    #c = cc(a,b).numpy()

    return x


def patch_image(target_image,patch,model,t_class,high,width):
    x_t_s, y_t_s, _ = target_image.shape
    x_p , y_p, _ = patch.shape
    x_p_r, y_p_r, _ = imutils.rotate_bound(patch,45).shape
    best_x, best_y, best_angle, prob_stop = 0,0,0,None
    jump = int(x_p/4)
    angle = 90
    best_pred = 50.0
    best_patch = None

    for m in range(0,350,angle):
        OTarget = target_image.copy()
        im = imutils.rotate_bound(patch,m)
        x,y,_ = im.shape
        part = OTarget[high:high+x,width:width+y]
        w = im == (0,0,0)
        im[w] = part[w]
        OTarget[high:high+x,width:width+y] = im
        OTarget = cv2.resize(OTarget,(30,30))
        OTarget = np.expand_dims(OTarget,0)
        pred = model.predict(OTarget)[0]
        close_n = calculate_diff(pred,t_class)
        if close_n < best_pred:
            best_pred = close_n
            best_x, best_y, best_angle = high, width, m
            best_patch = patch.copy()

    try:
        if best_patch == None:
            return best_pred, patch, best_pred, best_x, best_y, best_angle, prob_stop
    except:
        return best_pred, best_patch, best_pred, best_x, best_y, best_angle, prob_stop



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


def tournament(x1,x2):
    if x1 >= x2:
        return 1, 0
    else:
        return 0, 1

def mut_patch(target_image,patch,model,t_class,high,width):
    x_t_s, y_t_s, _ = target_image.shape
    x_p , y_p, _ = patch.shape
    x_p_r, y_p_r, _ = imutils.rotate_bound(patch,45).shape

    OTarget = target_image.copy()
    im = imutils.rotate_bound(patch,45)
    x,y,_ = im.shape
    part = OTarget[high:high+x,width:width+y]
    w = im == (0,0,0)
    im[w] = part[w]
    OTarget[high:high+x,width:width+y] = im
    OTarget = cv2.resize(OTarget,(30,30))
    OTarget = np.expand_dims(OTarget,0)
    pred = model.predict(OTarget)[0]
    close_n = calculate_diff(pred,t_class)
    best_pred = close_n
    best_patch = patch.copy()

    try:
        if best_patch == None:
            return best_pred, patch, best_pred
    except:
        return best_pred, best_patch, best_pred





def mutation(work_image,final_patches,n_close,model,t_class_arr,o,k):
    close_n, image, _,_,_,_,_ = patch_image(work_image,final_patches,model,t_class_arr,o,k)
    x,y,_ = image.shape
    patch_size = x
    if close_n < n_close:
        return image
    else:
        while True:
            for high in range(0,x-patch_size,int(patch_size/4)):
                for width in range(0,y-patch_size,int(patch_size/4)):
                    for j in range(1):
                        image_use = image.copy()
                        max_r = np.random.randint(0,255)
                        R = np.random.randint(max_r,255,(patch_size,patch_size)).astype(np.uint8)
                        G = np.random.randint(0,max_r,(patch_size,patch_size)).astype(np.uint8)
                        B = np.random.randint(0,max_r,(patch_size,patch_size)).astype(np.uint8)
                        patch = cv2.merge([B,G,R])
                        image_use[high:patch_size+high,width:patch_size+width] = patch
                        cv2.imshow("P",image_use)
                        cv2.waitKey(1)
                        close_n, image_use, _,_,_,_,_ = patch_image(work_image,image_use,model,t_class_arr,o,k)
                        if close_n < n_close:
                            return image_use
            patch_size -= 2



def genetic(target_image,model,t_class_arr):
    while True:
        try:
            form = input("\033[96mType of patches to apply (circle,rectangle,triangle,square) >\033[0m")
            x_target, _ = target_image.shape[0], target_image.shape[1]
            patch = initial_patch(target_image,form)
            cv2.imwrite(f"initial_patches/{form}.png",patch)
            break
        except:
            print("\033[91m     Please, enter a correct form. \033[0m")

    print("Initial patch has been saved!")
    patches = np.array([patch])
    sizes_patches = []
    for i in range(4,7,1):
        b = int(x_target/i)
        if b%2 == 1:
            b += 1
        sizes_patches.append(b)

    final_patches = []
    for i in sizes_patches:
        patch_ = []
        for j in patches:
            patch_.append(cv2.resize(j,(i,i)))
        final_patches.append(patch_)

    results_pred = []
    results_patch = []
    results_pos = []
    work_image = target_image.copy()
    for e in range(len(sizes_patches)):                          #Loop through different patches sizes
        x_t_s, y_t_s, _ = target_image.shape
        target_image = cv2.resize(work_image,(30,30))
        target_image = np.expand_dims(target_image,0)
        pred = model.predict(target_image)[0]
        close_n = calculate_diff(pred,t_class_arr)
        x_max, y_max, x_min, y_min = work_image.shape[0],work_image.shape[1],0,0
        x_p , y_p = sizes_patches[e],sizes_patches[e]
        jump = int(x_p/4)
        for o in range(0,max(x_min,x_max-x_p),jump):             #Loop through vertical side
            for k in range(0,max(y_min,y_max-y_p),jump):         #Loop through horizontal side
                print(f"\033[1m \033[4m \033[92m ____Generation:____\033[0m ")
                if close_n > 0.4:
                    final_patches_ = mutation(work_image,final_patches[e][0],close_n,model,t_class_arr,o,k)
                    final_patches[e] = final_patches_

                    print(f"Best result: {pred}")
                    close_n = pred

                    cv2.imwrite("best_temp_patch.png",final_patches[e][0])

                else:
                    cv2.imwrite(f"patches/p_sqare/final_patch{e}.png", results_patch[results_pred.index(min(results_pred))])
                    f = open("position_patch.csv","w")
                    writerID = csv.writer(f,lineterminator='\n')
                    writerID.writerow(results_pos[results_pred.index(min(results_pred))])






def all():
    while True:
        try:
            t_class = int(input("\033[96mTarget class (should be a number) >\033[0m"))
            break
        except:
            print("\033[91m     Please, enter a correct number. \033[0m")

    t_class_arr = np.zeros(43)
    t_class_arr[t_class] = 1
    folder_creation()
    model = keras.models.load_model("signs_classifier_model.h5")
    images = [cv2.imread(os.path.join(images_dir,i)) for i in os.listdir(images_dir)]
    images = images[1]
    genetic(images,model,t_class_arr)

if __name__ == "__main__":
    all()