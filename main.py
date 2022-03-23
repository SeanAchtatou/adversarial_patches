import random
import numpy as np
import simulator
import process_camera
import cv2
import apply_patch
import os
import time
from threading import Thread
from matplotlib import pyplot as plt

'''
Choose the form of the patches to apply on the image.
Deploy the simulation on BeamNG.

main > simulator
main > process_camera
main > apply_patch
main > simulator

'''
folders = ["images_detected","patch_images","patch_images_full"]
patches_folder = "patches"
patch_used = "patch.png"

def folder_creation():
    path = os.listdir()
    try:
        for i in folders:
            if i not in path:
                print(f"     Folder for the \033[4m{i}\033[0m \033[93m have been created.")
                os.mkdir(i)
            else:
                print(f"\033[93m     Folder for \033[4m{i}\033[0m \033[93m already exists.  \033[0m")

    except:
        print("\033[91m An error occured. \033[0m")

def patch_retrieve(form):
    path = os.listdir(patches_folder)
    try:
        for i in path:
            if i[2:5] in form:
                patches = os.listdir(f"{path}/{i}")
                patch = np.random.choice(patches)
                read_patch = cv2.imread(patch)
                cv2.imwrite(patch_used,read_patch)
                return read_patch
    except:
        print("\033[91mAn error occured.\033[0m")


if __name__ == "__main__":
    print("\033[6;30;47m\033[1m\033[4m########################### ADVERSARIAL PATCH 1.0 #############################\033[0m")
    forms_allowed = ["circle","rectangle","triangle","square"]
    while True:
        try:
            form = input("\033[7;30;44m[SYSTEM]\033[0m \033[1mType of patches to apply (circle,rectangle,triangle,square) >\033[0m")
            if form in forms_allowed:
                break
            else:
                print("\033[91m     Please insert a correct form!\033[0m")
        except:
            print("\033[91mAn unexpected error occured.\033[0m")
            exit()

    try:
        folder_creation()
        print("\033[92mRetrieving the patch... \033[0m")
        patch = patch_retrieve(form)
        print("\033[92mPatch retrieved. \033[0m")
    except:
        print("\033[1m \033[91m Unable to retrieve the patch! \033[0m")
        exit()

    print("\033[92mSimulator is being launched. \033[0m")
    print("\033[7;30;43m[Camera]\033[0m \033[96mObtaining the images from the camera...\033[0m ")
    for i in simulator.simulatorStart():
        full_image, pos, target_image, status = process_camera.image_pre_process(i)
        if status:
            f_image, state = apply_patch.patch_image(full_image, pos, target_image, patch)

            if state:
                print("Success!")
                simulator.send_instructions_to_car()
            else:
                print("Fail!")



