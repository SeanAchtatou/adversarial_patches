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


def patch_image(target_image,patch,model,t_class,x_max,y_max,x_min,y_min,high,width):
    x_t_s, y_t_s, _ = target_image.shape
    x_p , y_p, _ = patch.shape
    x_p_r, y_p_r, _ = imutils.rotate_bound(patch,45).shape
    best_x, best_y, best_angle, prob_stop = 0,0,0,None
    jump = int(x_p/4)
    angle = 90
    best_pred = 50.0
    best_patch = None

    """for i in range(50):
        random_x = np.random.randint(x_min,max(x_min+1,x_max-x_p))
        random_y = np.random.randint(y_min,max(y_min+1,y_max-y_p))
        OTarget = target_image.copy()
        im = imutils.rotate_bound(patch,0)
        x,y,_ = im.shape
        part = OTarget[random_x:random_x+x,random_y:random_y+y]
        w = im == (0,0,0)
        im[w] = part[w]
        OTarget[random_x:random_x+x,random_y:random_y+y] = im
        OTarget = cv2.resize(OTarget,(30,30))
        OTarget = np.expand_dims(OTarget,0)
        pred = model.predict(OTarget)[0]
        close_n = calculate_diff(pred,t_class)
        if close_n < best_pred:
            best_pred = close_n
            best_x, best_y, best_angle, prob_stop = random_x, random_y, 0, pred
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


    """for j in range(x_min,max(x_min,x_max-x_p_r),jump):
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
                    best_patch = patch.copy()"""
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


def genetic(target_image,model,t_class_arr):
    while True:
        try:
            form = input("\033[96mType of patches to apply (circle,rectangle,triangle,square) >\033[0m")
            numb_gen = 100
            population = 500
            tournament_k = 2
            best = population
            x_target, _ = target_image.shape[0], target_image.shape[1]

            patch = initial_patch(target_image,form)
            cv2.imwrite(f"initial_patches/{form}.png",patch)
            break
        except:
            print("\033[91m     Please, enter a correct form. \033[0m")

    print("Initial patch has been saved!")
    patches = np.array([patch])
    red_patch = [np.array(target_image[10:110,100:200]) for _ in range(5)]
    red_patch = np.squeeze(np.array([np.expand_dims(cv2.resize(i,(patch.shape[0],patch.shape[0])),0) for i in red_patch]),1)
    black_patch = np.zeros((5,patch.shape[0],patch.shape[0],3))
    white_patch = np.full((5,patch.shape[0],patch.shape[0],3),255)
    patches = np.append(patches,red_patch,axis=0)
    patches = np.append(patches,black_patch,axis=0)
    patches = np.append(patches,white_patch,axis=0)
    patches = np.append(patches,[initial_patch(target_image,form) for _ in range(1,population-15)],axis=0)

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
    final_patches_keep = final_patches.copy()
    for e in range(len(sizes_patches)):                          #Loop through different patches sizes
        x_t_s, y_t_s, _ = target_image.shape

        target_image = cv2.resize(work_image,(30,30))
        target_image = np.expand_dims(target_image,0)
        pred = model.predict(target_image)[0]
        close_n = calculate_diff(pred,t_class_arr)
        x_max, y_max, x_min, y_min = work_image.shape[0],work_image.shape[1],0,0
        last_close_n = close_n
        x_p , y_p = sizes_patches[e],sizes_patches[e]
        jump = int(x_p/4)
        for o in range(0,max(x_min,x_max-x_p),jump):             #Loop through vertical side
            for k in range(0,max(y_min,y_max-y_p),jump):         #Loop through horizontal side
                count = 0
                for i in range(numb_gen):                        #For the number of generation

                    if close_n == last_close_n:
                        last_close_n = close_n
                        count += 1

                    if count == 10:
                        final_patches = final_patches_keep.copy()
                        break

                    print(f"\033[1m \033[4m \033[92m ____Generation: {i}____\033[0m ")
                    if close_n > 0.4:
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
                            pred, image, close_n, best_x, best_y, best_a, prob_stop = patch_image(work_image,final_patches[e][j],model,t_class_arr, x_max,y_max,x_min,y_min,o,k)
                            results_pred.append(pred)
                            results_patch.append(image)
                            results_pos.append([best_x,best_y])
                            results_angle.append(best_a)
                            results_prob.append(prob_stop)

                        print(f"Best result: {min(results_pred)}")
                        close_n = min(results_pred)
                        #cv2.imshow("P",results_patch[results_pred.index(min(results_pred))])
                        #cv2.waitKey(0)

                        f_result_patch = []
                        f_result_angle = []
                        f_result_position = []
                        f_result_pred = []
                        cdf = []

                        """x = 0  #RANK SELECTION
                        for j in range(best):
                            x = results_pred.index(min(results_pred))
                            cdf.append(min(results_pred))
                            f_result_patch.append(results_patch[x])
                            f_result_position.append(results_pos[x])
                            f_result_angle.append(results_angle[x])
                            del results_pred[x]
                            del results_patch[x]"""

                         #TOURNAMENT SELECTION
                        results_patch_temp = results_patch.copy()
                        results_pred_temp = results_pred.copy()
                        results_pos_temp = results_pos.copy()
                        results_angle_temp = results_angle.copy()
                        for j in range(int(len(results_pred)/tournament_k)):
                            choices_p = np.random.randint(0,len(results_pred_temp),2)
                            choices_pred_1 = results_pred_temp[choices_p[0]]
                            choices_pred_2 = results_pred_temp[choices_p[1]]
                            final_r, toDel = tournament(choices_pred_1,choices_pred_2)
                            f_result_patch.append(results_patch_temp[choices_p[final_r]])
                            f_result_position.append(results_pos_temp[choices_p[final_r]])
                            f_result_angle.append(results_angle_temp[choices_p[final_r]])
                            f_result_pred.append(results_pred_temp[choices_p[final_r]])

                            del results_patch_temp[choices_p[0]]
                            del results_pred_temp[choices_p[0]]
                            del results_pos_temp[choices_p[0]]
                            del results_angle_temp[choices_p[0]]


                        results_pred = f_result_pred.copy()
                        results_patch = f_result_patch.copy()
                        final_patches[e] = f_result_patch
                        #print(cdf[:20])
                        #print(f_result_position[:20])
                        print(f_result_pred[:20])
                        close_n = f_result_pred[0]

                        f_result_position = np.array(f_result_position)
                        #x_max = f_result_position[f_result_position.argmax(axis=0)[0]][0]
                        #y_max = f_result_position[f_result_position.argmax(axis=0)[-1]][-1]
                        #x_min = f_result_position[f_result_position.argmin(axis=0)[0]][0]
                        #y_min = f_result_position[f_result_position.argmin(axis=0)[-1]][-1]
                        #print([x_max,y_max])
                        #print([x_min,y_min])
                        print(f_result_angle[:20])
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