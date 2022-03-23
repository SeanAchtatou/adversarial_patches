import cv2
import numpy as np

from matplotlib import pyplot as plt




def difference(im1,im2):
    b2,g2,r2 = cv2.split(im2)
    im2 = cv2.merge([r2,g2,b2])
    mean_r = int(np.mean(r2))
    mean_g = int(np.mean(g2))
    mean_b = int(np.mean(b2))

    threshold = [1,5,10,20,50,100,150,200]
    tt_other = [1,5,10,20,50,100,150,200]
    plt.figure()
    f, axarr = plt.subplots(8,8)

    countcount = 0
    for i in threshold:
        count = 0
        for j in tt_other:
            R = np.random.randint(max(0,mean_r-i),min(255,mean_r+i),(50,50)).astype(np.uint8)
            G = np.random.randint(max(0,mean_g-j),min(255,mean_g+j),(50,50)).astype(np.uint8)
            B = np.random.randint(max(0,mean_b-j),min(255,mean_b+j),(50,50)).astype(np.uint8)

            print(i,j)
            '''f, axar = plt.subplots(1,2)
            patch = cv2.merge([R,G,B])
            image[50:100,100:150] = patch
            axar[0].imshow(image)
            axar[0].axes.get_xaxis().set_visible(False)
            axar[0].axes.get_yaxis().set_visible(False)
            image[50:100,100:150] = im1
            axar[1].imshow(image)
            axar[1].axes.get_xaxis().set_visible(False)
            axar[1].axes.get_yaxis().set_visible(False)
            plt.show()'''

            patch = cv2.merge([R,G,B])
            axarr[count][countcount].imshow(patch)
            axarr[count][countcount].set_title(f"R:{i},G:{j},B:{j}",fontsize=4,color= 'blue', fontweight='bold')
            axarr[count][countcount].axes.get_xaxis().set_visible(False)
            axarr[count][countcount].axes.get_yaxis().set_visible(False)
            count += 1
        countcount += 1
    plt.subplots_adjust(left=0.125,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.9,
                        hspace=0.7)
    plt.show()


    return None


image = cv2.imread("images_/stop_sign_close.jpg")

patch = np.random.randint(0,255,(50,50,3))
apply_location = image[50:100,100:150].copy()
b,g,r = cv2.split(image)
image = cv2.merge([r,g,b])

difference(patch,apply_location)

