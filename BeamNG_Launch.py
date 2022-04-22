import random
import numpy as np
import BeamNG_Simulator
import BeamNG_Processcamera
import cv2
import BeamNG_Applypatch
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
folders = ["beamng_detected","beamng_patch"]
patches_folder = "patches"
patch_used = "BeamNG_patch.png"

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
                read_patch = cv2.imread(os.path.join(patches_folder,i,patch))
                cv2.imwrite(patch_used,read_patch)
                return read_patch

        else:
            print("No patch have been retrieved")
            exit()
    except:
        print("\033[91mAn error occured.\033[0m")


if __name__ == "__main__":
    print("\033[6;30;47m\033[1m\033[4m########################### BEAMNG WITH ADVERSARIAL PATCH SIMULATION #############################\033[0m")
    forms_allowed = ["circle","square"]
    while True:
        try:
            form = input("Type of patches to apply (circle,square) >")
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
    for i in BeamNG_Simulator.simulatorStart():
        full_image, pos, target_image, status = BeamNG_Processcamera.image_pre_process(i)
        if status:
            f_image, state = BeamNG_Applypatch.patch_image(full_image, pos, target_image, patch)

            if state:
                print("Success!")
                BeamNG_Simulator.send_instructions_to_car()
            else:
                print("Fail!")



