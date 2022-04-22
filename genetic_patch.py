import random
import os

import imutils
import time
import cv2
import csv
import numpy as np
import math
import mutation_patches
import tensorflow as tf

from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow import keras
from model import classes_



random.seed(1756)
patch_dir = "patches"
patch_images_dir = "patch_images"
initial_patch_dir = "initial_patches"
p_form = ["p_circle","p_square"]
images_dir = "images_"

def folder_creation():
    path = os.listdir()
    try:
        if patch_dir not in path:
            print("\033[93m     Folder for the \033[4m/patches\033[0m \033[93m have been created.\033[0m")
            os.mkdir(patch_dir)
            for i in p_form:
                os.mkdir(f"{patch_dir}/{i}")
        else:
            print("\033[93m     Folder for \033[4m/patches\033[0m \033[93m already exists.  \033[0m")


        if initial_patch_dir not in path:
            print("\033[93m     Folder for the \033[4m/initial_patches\033[0m \033[93m have been created.\033[0m")
            os.mkdir(initial_patch_dir)

        else:
            print("\033[93m     Folder for \033[4m/initial_patches\033[0m \033[93m already exists.  \033[0m")

        if patch_images_dir not in path:
            print("\033[93m     Folder for the \033[4m/patches_images\033[0m \033[93m have been created.\033[0m")
            os.mkdir(patch_images_dir)
        else:
            print("\033[93m     Folder for \033[4m/patches_images\033[0m \033[93m already exists.  \033[0m")

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

        return patch,0

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

        return patch,index


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


def patch_image(target_image,patch,model,t_class,x_max,y_max,x_min,y_min):
    x_t_s, y_t_s, _ = target_image.shape
    x_p , y_p, _ = patch.shape
    x_p_r, y_p_r, _ = imutils.rotate_bound(patch,45).shape
    best_x, best_y, best_angle, prob_stop = 0,0,0,None
    jump = int(x_p/4)
    angle = 90
    best_pred = 50.0
    best_patch = None

    for j in range(x_min,max(x_min,x_max-x_p_r),jump):
        for k in range(y_min,max(y_min,y_max-y_p_r),jump):
            for m in range(0,350,angle):
                OTarget = target_image.copy()
                im = imutils.rotate_bound(patch,m)
                x,y,_ = im.shape
                part = OTarget[j:j+x,k:k+y]
                w = im == (0,0,0)
                im[w] = part[w]
                OTarget[j:j+x,k:k+y] = im
                OTarget = cv2.resize(OTarget,(30,30))
                OTarget = np.expand_dims(OTarget,0)
                pred = model.predict(OTarget)[0]
                close_n = calculate_diff(pred,t_class)
                if close_n < best_pred:
                    best_pred = close_n
                    best_x, best_y, best_angle, prob_stop = j, k, m, pred
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


def mutation(patches):
    return mutation_patches.mutate(patches)

def final_patch_image(image,patch,pos,angle):
    OTarget = image.copy()
    x_pos, y_pos = pos
    im = imutils.rotate_bound(patch,angle)
    x,y,_ = im.shape
    part = OTarget[x_pos:x_pos+x,y_pos:y_pos+y]
    w = im == (0,0,0)
    im[w] = part[w]
    OTarget[x_pos:x_pos+x,y_pos:y_pos+y] = im
    cv2.imwrite(f"patches_images/final_patch_{pos[0]}X{pos[1]}Y_{angle}A_.png",OTarget)






def genetic(target_image,model,t_class_arr,no_target):
    while True:
        try:
            form = input("\nType of patches to apply (circle,square) >")
            numb_gen = 100
            population = 100
            tournament_k = 2
            best = population
            x_target, _ = target_image.shape[0], target_image.shape[1]

            patch, index = initial_patch(target_image,form)
            cv2.imwrite(f"initial_patches/{form}.png",patch)
            break
        except:
            print("\033[91m     Please, enter a correct form. \033[0m")
    print(f"\033[1;32;40m{form}\033[0m has been selected.")
    print("\033[32mInitial patch has been saved in \033[0m \033[0;30;42m/initial_patches\033[0m")
    patches = np.array([patch])
    color,r,g,b = mean(target_image)
    R_channel = np.full((patch.shape[0],patch.shape[0]),r)
    G_channel = np.full((patch.shape[0],patch.shape[0]),g)
    B_channel = np.full((patch.shape[0],patch.shape[0]),b)

    if form == "circle":
        R_channel[:,:][index] = 0
        G_channel[:,:][index] = 0
        B_channel[:,:][index] = 0
        patch_color = cv2.merge([B_channel,G_channel,R_channel])
    else:
        patch_color = cv2.merge([B_channel,G_channel,R_channel])

    color_patch = [patch_color for _ in range(5)]
    color_patch = np.squeeze(np.array([np.expand_dims(cv2.resize(i,(patch.shape[0],patch.shape[0])),0) for i in color_patch]),1)
    black_patch = np.zeros((5,patch.shape[0],patch.shape[0],3))
    patches = np.append(patches,color_patch,axis=0)
    patches = np.append(patches,black_patch,axis=0)
    patches = np.append(patches,[initial_patch(target_image,form)[0] for _ in range(1,population-10)],axis=0)

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
    results_angle = []
    work_image = target_image.copy()
    for e in range(len(sizes_patches)):                          #Loop through different patches sizes
        x_t_s, y_t_s, _ = target_image.shape
        successfull = False
        target_image = cv2.resize(work_image,(30,30))
        target_image = np.expand_dims(target_image,0)
        pred = model.predict(target_image)[0]
        value_higher = np.argmax(pred)
        if no_target:
            t_class_arr[value_higher] = 0
        close_n = calculate_diff(pred,t_class_arr)
        x_max, y_max, x_min, y_min = work_image.shape[0],work_image.shape[1],0,0

        for i in range(numb_gen):                        #For the number of generation
            print(f"\033[1m \033[4m \033[92m ____Generation: {i}____\033[0m ")
            if (close_n > 0.4) and (successfull != True):
                results_pred = []
                results_patch = []
                results_pos = []
                results_angle = []
                results_prob = []
                np.random.shuffle(final_patches[e])
                final_patches_ = mutation(final_patches[e])
                final_patches[e] = np.append(final_patches[e],final_patches_,axis=0)
                np.random.shuffle(final_patches[e])

                for j in tqdm(range(len(final_patches[e]))):
                    pred, image, close_n, best_x, best_y, best_a, prob_stop = patch_image(work_image,final_patches[e][j],model,t_class_arr, x_max,y_max,x_min,y_min)
                    results_pred.append(pred)
                    results_patch.append(image)
                    results_pos.append([best_x,best_y])
                    results_angle.append(best_a)
                    results_prob.append(prob_stop)

                print(f"Best result: {min(results_pred)}")
                #cv2.imshow("P",results_patch[results_pred.index(min(results_pred))])
                #cv2.waitKey(0)

                f_result_patch = []
                f_result_angle = []
                f_result_position = []
                f_result_pred = []
                f_result_prob = []
                cdf = []

                x = 0  #RANK SELECTION
                """for j in range(best):
                    x = results_pred.index(min(results_pred))
                    cdf.append(min(results_pred))
                    f_result_patch.append(results_patch[x])
                    f_result_position.append(results_pos[x])
                    f_result_pred.append(results_pred[x])
                    f_result_angle.append(results_angle[x])
                    f_result_stop.append(results_prob[x])
                    del results_pred[x]
                    del results_patch[x]
                    del results_pos[x]
                    del results_angle[x]
                    del results_prob[x]"""

                 #TOURNAMENT SELECTION
                results_patch_temp = results_patch.copy()
                results_pred_temp = results_pred.copy()
                results_pos_temp = results_pos.copy()
                results_angle_temp = results_angle.copy()
                results_prob_temp = results_prob.copy()
                for j in range(int(len(results_pred)/tournament_k)):
                    choices_p = np.random.randint(0,len(results_pred_temp),2)
                    choices_pred_1 = results_pred_temp[choices_p[0]]
                    choices_pred_2 = results_pred_temp[choices_p[1]]
                    final_r, toDel = tournament(choices_pred_1,choices_pred_2)
                    f_result_patch.append(results_patch_temp[choices_p[final_r]])
                    f_result_position.append(results_pos_temp[choices_p[final_r]])
                    f_result_angle.append(results_angle_temp[choices_p[final_r]])
                    f_result_pred.append(results_pred_temp[choices_p[final_r]])
                    f_result_prob.append(results_prob_temp[choices_p[final_r]])

                    del results_patch_temp[choices_p[0]]
                    del results_pred_temp[choices_p[0]]
                    del results_pos_temp[choices_p[0]]
                    del results_angle_temp[choices_p[0]]
                    del results_prob_temp[choices_p[0]]


                results_pred = f_result_pred.copy()
                results_patch = f_result_patch.copy()
                results_angle = f_result_angle.copy()
                results_pos = f_result_position.copy()
                final_patches[e] = f_result_patch

                print(f_result_pred[:5])
                print(f_result_position[:5])
                print(f_result_prob[:5])

                close_n = f_result_pred[0]
                if np.argmax(close_n) != value_higher:
                    print("\033[32mSuccess!\033[0m]")
                    print(f"     Predicted \033[1;32;40m{classes_[np.argmax(close_n)]}\033[0m with probability \033[1;32;40m{close_n[np.argmax(close_n)]}\033[0m")
                    print(f"     Predicted \033[1;32;40m{classes_[value_higher]}\033[0m with probability \033[1;32;40m{close_n[value_higher]}\033[0m")

                    successfull = True
                #f_result_position = np.array(f_result_position)
                #x_max = f_result_position[f_result_position.argmax(axis=0)[0]][0]
                #y_max = f_result_position[f_result_position.argmax(axis=0)[-1]][-1]
                #x_min = f_result_position[f_result_position.argmin(axis=0)[0]][0]
                #y_min = f_result_position[f_result_position.argmin(axis=0)[-1]][-1]
                #print([x_max,y_max])
                #print([x_min,y_min])
                print(f_result_angle[:5])
                cv2.imwrite("best_temp_patch.png",final_patches[e][0])

            else:
                pos = results_pos[results_pred.index(min(results_pred))]
                angle = results_angle[results_pred.index(min(results_pred))]
                cv2.imwrite(f"patches/p_{form}/final_patch{sizes_patches[e]}_{pos[0]}X{pos[1]}Y_{angle}A_.png",results_patch[results_pred.index(min(results_pred))])
                final_patch_image(target_image,results_patch[results_pred.index(min(results_pred))],pos,angle)
                f = open("position_patch.csv","w")
                writerID = csv.writer(f,lineterminator='\n')
                writerID.writerow(results_pos[results_pred.index(min(results_pred))])






def all():
    print("\033[1;33;40m ################################# ADVERSARIAL PATCH WITH COLOR RESTRICTION #################################### \033[0m \n")

    images = [i for i in os.listdir(images_dir)]
    if len(images) < 1:
        print(f"\033[91m     None image have been found. Please insert an image in the corresponding folder {images_dir}. \033[0m")
        exit()
    else:
        print("\033[4m\033[2;37;35mImages found:\033[0m")
        count = 0
        for i in images:
            print(f"    [\033[32m{count}\033[0m] {i}")

    while True:
        try:
            image_selected = int(input("\nEnter the number of the Image to attack (should be a number) >"))
            image = cv2.imread(os.path.join(images_dir,images[image_selected]))
            print(f"\033[1;32;40m{images[image_selected]}\033[0m has been selected.")
            break
        except:
            print("\033[91m     Please, enter a correct number. \033[0m")

    while True:
        try:
            t_class = input("\nTarget class or not(should be a number or empty for no target ) >")
            if t_class == "":
                print(f"\033[1;32;40mNo Target\033[0m has been selected.")
                t_class_arr = np.ones(43)
                no_target = True
            else:
                print(f"\033[1;32;40m{classes_[int(t_class)+1]}\033[0m has been selected.")
                t_class_arr = np.zeros(43)
                t_class_arr[int(t_class)] = 1
                no_target = False
            break
        except:
            print("\033[91m     Please, enter a correct number. \033[0m")

    folder_creation()
    model = keras.models.load_model("signs_classifier_model.h5")
    genetic(image,model,t_class_arr,no_target)

if __name__ == "__main__":
    all()