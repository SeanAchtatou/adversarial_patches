import numpy as np
import tensorflow as tf
from tensorflow import keras
from pymoo.problems.multi import Problem,ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import cv2
import os
import imutils
import time

siye = 2
t_class_arr = np.zeros(siye)
t_class_arr[1] = 1

def calculate_diff(a,b):
    """x = 0
    count = 0
    for i in a:
        x += (i-b[count])**2
        count += 1"""
    cc = tf.keras.metrics.CategoricalCrossentropy()
    c = cc(a,b).numpy()
    return c

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=[0 for _ in range(siye)],
                         xu=[1 for _ in range(siye)])

    def _evaluate(self, X, out, *args, **kwargs):
        f2 = []
        x = time.time()
        for i in X:
            f2.append(((i[0] - 0) **2) + ((i[1] - 1)**2))
        y = time.time()
        print(y-x)

        f1 = (((X[:,0] - 0) **2) + ((X[:,1] - 1)**2))
        out["F"] = np.column_stack([f2])



vectorized_problem = MyProblem()
algorithm = GA(
    pop_size=200,
    n_offsprings=50,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 100)

res = minimize(vectorized_problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
