import cv2
import numpy as np

model = cv2.dnn.readNetFromCaffe("deploy.prototxt","model_road_signs.caffemodel")

image = cv2.imread("stop_sign.jpg")

image = cv2.resize(image,(227,227))
image = cv2.dnn.blobFromImage(image)

print(image.shape)
model.setInput(image)
output = model.forward()
print(np.argmax(output))
