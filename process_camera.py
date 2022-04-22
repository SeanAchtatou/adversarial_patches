import random
import numpy as np
import cv2
import time
import simulator, evaluation_apply_patch
from tensorflow import keras


'''
Take the image from the car and check with a model if a sign is detected.
If a sign is detected, start the proprecc of the original image. We obtain a mask of the original image.
Moreover, we take the part focused of the image for further details.
Else, nothing happens.
'''
model = keras.models.load_model("signs_classifier_model.h5")
masks = [500,250,100]

def image_pre_process(image):
    status = False
    best_pred = 0
    im_f = None
    position = None

    imageO = image.copy()

    x, y, _ = image.shape
    for k in masks:
        for i in range(0,y-k,100):
            for j in range(0,x-k,100):
                image = imageO[j:j+k,i:i+k]
                temp_im = image.copy()
                image = cv2.resize(image,(30,30))
                image = np.expand_dims(image, axis=0)
                image = np.array(image)
                pred = model.predict(image)[0]
                class_nb = np.argmax(pred)
                if (class_nb == 14) and (max(pred) >= 0.7) and (max(pred) > best_pred):
                    best_pred = max(pred)
                    position = (i,j)
                    im_f = temp_im.copy()
                    status = True

    cv2.imshow("T",im_f)
    cv2.waitKey(0)
    cv2.imwrite(f"images_detected/target_{str(time.time()*1000)[:10]}.png",im_f)
    cv2.imwrite(f"images_detected/original_{str(time.time()*1000)[:9]}.png",imageO)
    return imageO, position, im_f, status


def image_patch_postprocess(image):
    image = cv2.resize(image,(30,30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict(image)[0]
    class_nb = np.argmax(pred)
    if (class_nb == 6) and (max(pred) >= 0.5):
        return max(pred), True
    return 0,False

if __name__ == "__main__":
    image = cv2.imread("stop_sign.jpg")
    image, pos, target_im, status = image_pre_process(image)
    y = time.time()
    evaluation_apply_patch.patch_image(image, pos, target_im, status)
    x = time.time()
    print(x-y)


