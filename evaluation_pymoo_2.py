import numpy as np
import tensorflow as tf
import cv2
import os
import imutils
import random

from tensorflow import keras
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
p_form = ["p_circle","p_rect","p_tri","p_sqr"]
images_dir = "images_"
color_main = None

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

    if (mean_r >= (mean_g and mean_b)):
        return "R",mean_r,mean_g,mean_b
    if (mean_g >= (mean_b and mean_r)):
        return "G",mean_r,mean_g,mean_b
    if (mean_b >= (mean_g and mean_r)):
        return "B",mean_r,mean_g,mean_b


def calculate_diff(a,b):
    """x = 0
    count = 0
    for i in a:
        x += (i-b[count])**2
        count += 1"""

    cc = tf.keras.metrics.CategoricalCrossentropy()
    c = cc(a,b).numpy()
    return c

def Algorithm(size_patch,height,width,angle,t_class,model):
    n_var = (size_patch**2)*3

    class MyProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=n_var,
                             n_obj=1,
                             n_constr=0,
                             xl=[0 for i in range(n_var)],
                             xu=[255 if ((i+1)%3)==0 else 150 for i in range(n_var)])

        def _evaluate(self, X, out, *args, **kwargs):
            OTarget = images.copy()
            X = np.reshape(X,(size_patch,size_patch,3)).astype(np.uint8)

            b_m,g_m,r_m = mean_patch(X)
            im = imutils.rotate_bound(X,angle)
            x,y,_ = im.shape
            part = OTarget[height:height+x,width:width+y]
            w = im == (0,0,0)
            im[w] = part[w]
            OTarget[height:height+x,width:width+y] = im
            OTarget = cv2.resize(OTarget,(30,30))
            OTarget = np.expand_dims(OTarget,0)
            pred = model.predict(OTarget)[0]
            f1 = calculate_diff(pred,t_class)

            ep = 0.9
            if color_main == "R":
                #g1 = b_m - (ep*r_m)
                #g2 = g_m - (ep*r_m)
                g1 = b_m - 100
                g2 = g_m - 100
            if color_main == "G":
                g1 = r_m - (ep*g_m)
                g2 = b_m - (ep*g_m)
            if color_main == "B":
                g1 = r_m - (ep*b_m)
                g2 = g_m - (ep*b_m)

            out["F"] = np.column_stack([f1])
            #out["G"] = np.column_stack([g1, g2])


    """class MyProblemMulti(Problem):
        def __init__(self):
            super().__init__(n_var=n_var,
                             n_obj=1,
                             n_constr=0,
                             xl=[0 for _ in range(n_var)],
                             xu=[255 if ((i+1)%3)==0 else 150 for i in range(n_var)])

        def _evaluate(self, X, out, *args, **kwargs):
            f1 = []
            for i in X:
                OTarget = images.copy()
                i = np.reshape(i,(size_patch,size_patch,3)).astype(np.uint8)

                b_m,g_m,r_m = mean_patch(i)
                im = imutils.rotate_bound(i,angle)
                x,y,_ = im.shape
                part = OTarget[height:height+x,width:width+y]
                w = im == (0,0,0)
                im[w] = part[w]
                OTarget[height:height+x,width:width+y] = im
                OTarget = cv2.resize(OTarget,(30,30))
                OTarget = np.expand_dims(OTarget,0)
                pred = model.predict(OTarget)[0]
                f1.append(calculate_diff(pred,t_class))

                ep = 0.9
                if color_main == "R":
                    #g1 = b_m - (ep*r_m)
                    #g2 = g_m - (ep*r_m)
                    g1 = b_m - 100
                    g2 = g_m - 100
                if color_main == "G":
                    g1 = r_m - (ep*g_m)
                    g2 = b_m - (ep*g_m)
                if color_main == "B":
                    g1 = r_m - (ep*b_m)
                    g2 = g_m - (ep*b_m)

            out["F"] = np.column_stack([f1])
            #out["G"] = np.column_stack([g1, g2])"""

    vectorized_problem = MyProblem()
    algorithm = GA(
        pop_size=100,
        n_offsprings=200,
        selection=get_selection("random"),
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=0.9, eta=15),
        mutation=get_mutation("int_pm", eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 200)

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
        image = np.reshape(image_flat,(size_patch,size_patch,3)).astype(np.uint8)
        cv2.imshow("P",image)
        cv2.waitKey(1)
    except:
        print("No solution found here")


def main(image,model,t_class):
    global color_main
    jump = 5
    angle = 5
    x,y,z = image.shape
    color, mean_r, mean_g, mean_b = mean(images)
    color_main = color
    sizes_patches = []
    for i in range(3,9,2):
        b = int(x/i)
        if b%2 == 1:
            b += 1
        sizes_patches.append(b)

    for i in sizes_patches:
        for m in range(0,350,angle):
            for j in range(0,x-i,jump):
                for k in range(0,y-i,jump):
                    Algorithm(i,j,k,m,t_class,model)


if "__main__" == __name__:
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
    main(images,model,t_class_arr)
