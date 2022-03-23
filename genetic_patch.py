import random
import os


import cv2
import csv
import numpy as np
import math
import mutation_patches

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


def initial_patch(image,form):
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

    return x


def patch_image(target_image,patch,model,t_class):
    x_t_s, y_t_s, _ = target_image.shape
    x_p , y_p, _ = patch.shape
    best_x, best_y = 0,0
    jump = 100

    best_pred = 2.0
    best_patch = None

    """for i in range(5):
        random_x = np.random.randint(0,x_t_s-x_p)
        random_y = np.random.randint(0,y_t_s-y_p)
        OTarget = target_image.copy()
        OTarget[random_x:random_x+x_p,random_y:random_y+y_p] = patch
        OTarget = cv2.resize(OTarget,(30,30))
        OTarget = np.expand_dims(OTarget,0)
        pred = model.predict(OTarget)[0]
        close_n = calculate_diff(pred,t_class)
        if close_n < best_pred:
            best_pred = close_n
            best_x, best_y = random_x, random_y
            OTarget = np.squeeze(OTarget,0)
            best_patch = patch.copy()"""


    """for j in range(0,int((x_t_s-x_p)/2),jump):
        for k in range(0,int((y_t_s-y_p)/2),jump):
            OTarget, OTarget2, OTarget3, OTarget4 = target_image.copy(),target_image.copy(),target_image.copy(),target_image.copy()
            OTargets = [OTarget,OTarget2,OTarget3,OTarget4]
            OTarget[j:j+x_p,k:k+y_p] = patch
            OTarget[j:j+x_p,y_t_s-y_p-k:y_t_s-k] = patch
            OTarget[x_t_s-x_p-j:x_t_s-j,k:k+y_p] = patch
            OTarget[x_t_s-x_p-j:x_t_s-j,y_t_s-y_p-k:y_t_s-k] = patch
            #cv2.imshow("P",OTarget)
            #cv2.waitKey(0)
            preds = []
            for m in OTargets:
                m = cv2.resize(m,(30,30))
                m = np.expand_dims(m,0)
                pred = model.predict(m)[0]
                close_n = calculate_diff(pred,t_class)
                preds.append(close_n)
            x = preds[np.argmin(preds)]
            if x < best_pred:
                best_pred = x
                best_x, best_y = j, k
                best_patch = patch.copy()"""


    for j in range(0,x_t_s-x_p,jump):
        for k in range(0,y_t_s-y_p,jump):
            OTarget = target_image.copy()
            OTarget[j:j+x_p,k:k+y_p] = patch
            OTarget = cv2.resize(OTarget,(30,30))
            OTarget = np.expand_dims(OTarget,0)
            pred = model.predict(OTarget)[0]
            close_n = calculate_diff(pred,t_class)
            if close_n < best_pred:
                best_pred = close_n
                best_x, best_y = j, k
                best_patch = patch.copy()
    try:
        if best_patch == None:
            return best_pred, patch, best_pred, best_x, best_y
    except:
        return best_pred, best_patch, best_pred, best_x, best_y



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




def mutation(patches):
    return mutation_patches.mutate(patches)


def genetic(target_image,model,t_class_arr):
    form = input("Type of patches to apply (circle,rectangle,triangle,square) >")
    numb_gen = 100
    population = 100
    best = population
    x_target, _ = target_image.shape[0], target_image.shape[1]

    patch = initial_patch(target_image,form)
    cv2.imwrite(f"initial_patches/{form}.png",patch)
    print("Initial patch has been saved!")
    patches = np.array([patch])
    patches = np.append(patches,[initial_patch(target_image,form) for _ in range(1,population)],axis=0)

    sizes_patches = []
    for i in range(5,11,2):
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



    """final_patches = []
    size = 0
    for i in range(len(patches)):
        if i % len(sizes_patches) == 0:
            size = 0
        final_patches.append(cv2.resize(patches[i],(sizes_patches[size],sizes_patches[size])))
        size += 1

    for i in range(len(final_patches)):
        size_patch = max(sizes_patches)
        x,y,_ = final_patches[i].shape
        v = int((size_patch-x)/2)
        final_patches[i] = np.pad(final_patches[i],[(v,v),(v,v),(0,0)],"constant")"""


    work_image = target_image.copy()
    target_image = cv2.resize(target_image,(30,30))
    target_image = np.expand_dims(target_image,0)
    pred = model.predict(target_image)[0]
    close_n = calculate_diff(pred,t_class_arr)
    results_pred = []
    results_patch = []
    results_pos = []
    for e in range(len(sizes_patches)):
        close_n = calculate_diff(pred,t_class_arr)
        for i in range(numb_gen):
            print(f"\033[1m \033[4m \033[92m ____Generation: {i}____\033[0m ")

            if close_n > 0.4:
                results_pred = []
                results_patch = []
                results_pos = []
                np.random.shuffle(final_patches[e])
                final_patches_ = mutation(final_patches[e])
                final_patches[e] = np.append(final_patches[e],final_patches_,axis=0)
                np.random.shuffle(final_patches[e])

                for j in tqdm(range(len(final_patches[e]))):
                #for j in final_patches:
                    pred, image, close_n, best_x, best_y = patch_image(work_image,final_patches[e][j],model,t_class_arr)
                    results_pred.append(pred)
                    results_patch.append(image)
                    results_pos.append([best_x,best_y])

                print(min(results_pred))
                close_n = min(results_pred)
                #cv2.imshow("P",results_patch[results_pred.index(min(results_pred))])
                #cv2.waitKey(0)

                f_result_patch = []
                for j in range(best):
                    x = results_pred.index(min(results_pred))
                    f_result_patch.append(results_patch[x])
                    del results_pred[x]
                    del results_patch[x]

                final_patches[e] = f_result_patch

                """f_result_patch = []
                size = 0
                for i in range(len(final_patches)):
                    if i % 4 == 0:
                        size = 0
                    f_result_patch.append(cv2.resize(final_patches[i],(sizes_patches[size],sizes_patches[size])))
                    size += 1"""

                """final_patches = f_result_patch"""
            else:
                cv2.imwrite(f"final_patch{e}.png",results_patch[results_pred.index(min(results_pred))])
                f =  open("position_patch.csv","w")
                writerID = csv.writer(f,lineterminator='\n')
                writerID.writerow(results_pos[results_pred.index(min(results_pred))])






def all():
    t_class = int(input("Target class (should be a number) >>"))
    t_class_arr = np.zeros(43)
    t_class_arr[t_class] = 1
    folder_creation()
    model = keras.models.load_model("signs_classifier_model.h5")
    images = [cv2.imread(os.path.join(images_dir,i)) for i in os.listdir(images_dir)]
    images = images[1]
    genetic(images,model,t_class_arr)

if __name__ == "__main__":
    all()