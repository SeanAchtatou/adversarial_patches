import numpy as np
import random
import math


a = np.array([0.1,0.2,0.6,0.1])

b = np.zeros(5)
b[4] = 1


def both(a,b):
    x = 0
    count = 0
    for i in a:
        x += (i-b[count])**2
        count += 1
    print(x)


def evaluate_own(a):
    x = 0
    for i in a:
        x += math.exp(i)
    x -= a[1]
    x = math.log(x)

    print(x)

while True:
    a = np.array([0.2,0.3,0.4,0.1])
    print(f"Best probability position : {np.argmax(a)}")
    print(f"Best probability : {a[np.argmax(a)]}")
    print(f"Target prob class prob : {a[2]}")
    evaluate_own(b)
    #both(a,b)
    input("W.")
#both(a,b)

