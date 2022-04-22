import random
import os

import cv2
import numpy as np
import process_camera
from tensorflow import keras

import math

model = keras.models.load_model("signs_classifier_model.h5")
patch_images = "patch_images"
patch_images_f = "patch_images_full"

def patch_image(image,pos,target_image,patch):
    x_t_s, y_t_s, _ = target_image.shape
    patches_sizes = [np.resize(patch,(int(x_t_s/i),int(y_t_s/i),3)) for i in range(5,10)]
    image_p = image.copy()

    pred = 0
    f_im = None
    state = False
    OTarget = None
    for i in patches_sizes:
        for j in range(0,x_t_s-i.shape[0],30):
            for k in range(0,y_t_s-i.shape[0],30):
                OTarget = target_image.copy()
                OTarget[j:j+i.shape[0],k:k+i.shape[1]] = i
                pred_, status = process_camera.image_patch_postprocess(OTarget)
                if status:
                    if pred_ > pred:
                        pred = pred_
                        f_im = OTarget
                        state = True

    if state:
        image_p[pos[1]:pos[1]+y_t_s,pos[0]:pos[0]+x_t_s] = OTarget
        cv2.imwrite(f"{patch_images}/original_t.png",target_image)
        cv2.imwrite(f"{patch_images}/result_t.png",f_im)
        cv2.imwrite(f"{patch_images_f}/original_f.png",image)
        cv2.imwrite(f"{patch_images_f}/result_f.png",image_p)

    return f_im, state


