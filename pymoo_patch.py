import numpy as np
import tensorflow as tf
import cv2
import os
import imutils
import random
import time

from tensorflow import keras
from model import classes_
from pymoo.problems.multi import Problem,ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_selection, get_termination
from pymoo.optimize import minimize

random.seed(1756)
patch_dir = "patches"

p_form = ["p_circle","p_square","p_image"]
images_dir = "images_"
color_main = None
t_class_number = 0

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

def mean_patch(image):
    b,g,r = cv2.split(image)
    mean_r = int(np.mean(r))
    mean_g = int(np.mean(g))
    mean_b = int(np.mean(b))

    return mean_b,mean_g,mean_r

def mean(image):
    b,g,r = cv2.split(image)
    mean_r = int(np.mean(r))
    mean_g = int(np.mean(g))
    mean_b = int(np.mean(b))

    if (mean_r >= mean_g) and (mean_r >= mean_b):
        return "R",mean_r,mean_g,mean_b
    if (mean_g >= mean_b) and (mean_g >= mean_r):
        return "G",mean_r,mean_g,mean_b
    if (mean_b >= mean_g) and (mean_b >= mean_r):
        return "B",mean_r,mean_g,mean_b


def calculate_diff(a,b):
    x = []
    for i in a:
        x.append(np.sum((i-b)**2))

    #cc = tf.keras.metrics.CategoricalCrossentropy()
    #c = [cc(i,b).numpy() for i in a]
    return x

def Algorithm(size_patch,t_class,model,image,form):
    original_prob = np.argmax(model.predict(np.expand_dims(cv2.resize(image,(30,30)),0)))
    x,y,z = image.shape
    n_var = ((size_patch**2)*3)+3
    last_ = (size_patch**2)*3

    patch_temp = np.reshape([0 for _ in range(last_)],(size_patch,size_patch,3)).astype(np.uint8)
    x_max,y_max,_ = imutils.rotate_bound(patch_temp,45).shape

    if color_main == "R":
        channel_c = 3
    if color_main == "G":
        channel_c = 2
    if color_main == "B":
        channel_c = 1

    x_l = [200 if ((i+1)%channel_c)==0 else 0 for i in range(n_var)]
    x_l[-1] = 0
    x_l[-2] = 0
    x_l[-3] = 0
    x_u = []
    for i in range(n_var):
        if i == (last_):
            x_u.append(x-x_max)
        if i == (last_+1):
            x_u.append(y-y_max)
        if i == (last_+2):
            x_u.append(360)
        if (i < last_):
            if ((i+1)%channel_c)==0:
                x_u.append(255)
            else:
                x_u.append(100)


    class MyProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n_var,
                             n_obj=1,
                             n_constr=0,
                             xl=x_l,
                             xu=x_u)

        def _evaluate(self, X, out, *args, **kwargs):
            patches_ = X[:,:-3]
            heights_ = X[:,-3]
            widths_ = X[:,-2]
            angles_ = X[:,-1]
            patches = np.reshape(patches_,(len(X),size_patch,size_patch,3)).astype(np.uint8)

            if form == "circle":
                radius = patches.shape[1]/2
                y, x = np.ogrid[-radius: radius, -radius: radius]
                index = x**2 + y**2 > radius**2
                for i in range(len(patches)):
                    patches[i][:,:,:][index] = 0

            final_image = []
            #b_set = []
            #g_set = []
            #r_set = []
            #t1 = time.time()
            for i in range(len(patches)):
                #b_, g_, r_ = mean_patch(patches[i])
                #b_set.append(b_)
                #g_set.append(g_)
                #r_set.append(r_)
                OTarget = image.copy()
                im = imutils.rotate_bound(patches[i],angles_[i])
                x,y,_ = im.shape
                part = OTarget[heights_[i]:heights_[i]+x,widths_[i]:widths_[i]+y]
                w = im == (0,0,0)
                im[w] = part[w]
                OTarget[heights_[i]:heights_[i]+x,widths_[i]:widths_[i]+y] = im
                cv2.imshow("P",OTarget)
                cv2.waitKey(1)
                OTarget = cv2.resize(OTarget,(30,30))
                final_image.append(OTarget)

            pred = model.predict(np.array(final_image))
            print(f"Class predicted: \n {pred.argmax(1)[:5]}")
            print(f"Probability of highest class: \n{pred[np.array([i for i in range(len(pred))][:5]),pred.argmax(1)[:5]]}")
            print(f"Probability of the class of the original image: \n{pred[:,original_prob][:5]}")
            f1 = calculate_diff(pred,t_class)

            """b_set = np.array(b_set)
            g_set = np.array(g_set)
            r_set = np.array(r_set)
            eps = 0.5
            if color_main == "R":
                g1_ = b_set - (eps*r_set)
                g2_ = g_set - (eps*r_set)
            if color_main == "G":
                g1_ = r_set - g_set
                g2_ = b_set - g_set
            if color_main == "B":
                g1_ = r_set - b_set
                g2_ = g_set - b_set"""

            #t2 = time.time()
            #print(f"Time: {t2-t1}")

            out["F"] = np.column_stack([f1])
            #out["G"] = np.column_stack([g1_, g2_])

    vectorized_problem = MyProblem()
    pop_size = 300
    algorithm = GA(
        pop_size=pop_size,
        n_offsprings=100,
        selection=get_selection("random"),
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=0.9, eta=15),
        mutation=get_mutation("int_pm", eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 500)

    res = minimize(vectorized_problem,
                   algorithm,
                   termination,
                   seed=3746,
                   save_history=True,
                   verbose=True,
                   return_least_infeasible=True)

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    try:
        image_flat = res.X
        patch_ = image_flat[:-3]
        height_ = image_flat[-3]
        width_ = image_flat[-2]
        angle_ = image_flat[-1]
        patch = np.reshape(patch_,(size_patch,size_patch,3)).astype(np.uint8)

        if form == "circle":
            radius = patch.shape[0]/2
            y, x = np.ogrid[-radius: radius, -radius: radius]
            index = x**2 + y**2 > radius**2
            patch[:,:,:][index] = 0

        OTarget = image.copy()
        im = imutils.rotate_bound(patch,angle_)
        x,y,_ = im.shape
        part = OTarget[height_:height_+x,width_:width_+y]
        w = im == (0,0,0)
        im[w] = part[w]
        OTarget[height_:height_+x,width_:width_+y] = im
        cv2.imwrite(f"{patch_dir}/p_{form}/final_patch_{size_patch}S_{height_}X{width_}Y_{angle_}A_{int(time.time())}T.png",patch)
        cv2.imwrite(f"{patch_dir}/{p_form[2]}/patch_image_{size_patch}S_{height_}X{width_}Y_{angle_}A_{int(time.time())}T.png",OTarget)
        cv2.imshow(f"Best patch for size-{size_patch}",patch)
        cv2.imshow(f"Position of best patch in Image",OTarget)
        cv2.waitKey(1)
    except:
        print("No solution found here.")


def main(image,model,t_class,form):
    global color_main
    x,y,z = image.shape
    color, mean_r, mean_g, mean_b = mean(image)
    color_main = color
    sizes_patches = []
    for i in range(3,8,1):
        b = int(x/i)
        if b%2 == 1:
            b += 1
        sizes_patches.append(b)

    for i in sizes_patches:
        Algorithm(i,t_class,model,image,form)


if "__main__" == __name__:
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
            count += 1

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
            print(f"Road Signs classes: \n {classes_}")
            t_class = input("\nTarget class or not(should be a number or empty for no target ) >")
            if t_class == "":
                print(f"\033[1;32;40mNo Target\033[0m has been selected. Random one chosen.")
                t_class_arr = np.zeros(43)
                t_class = np.random.randint(0,43)
                t_class_arr[int(t_class)] = 1
                t_class_number = int(t_class)
                no_target = True
            else:
                print(f"\033[1;32;40m{classes_[int(t_class)+1]}\033[0m has been selected.")
                t_class_arr = np.zeros(43)
                t_class_arr[int(t_class)] = 1
                t_class_number = int(t_class)
                no_target = False
            break
        except:
            print("\033[91m     Please, enter a correct number. \033[0m")

    while True:
        try:
            form = input("\nType of patches to apply (circle,square) >")
            break
        except:
            print("\033[91m     Please, enter a correct form. \033[0m")

    print(f"\033[1;32;40m{form}\033[0m has been selected.")

    folder_creation()
    model = keras.models.load_model("signs_classifier_model.h5")
    main(image,model,t_class_arr,form)

#For untargeted attack, modify in order to retrieve the best one with best probabilities
#, modify the array compared, respective to the best class.
#For targetd attack, check if at the end, we have the resepctive class target, if yes, provide
#the probabilities obtained, and the image, and patch. If not, provide which class is the best
#with the image and the patch. Thus, we should re-run the algorithm with it.