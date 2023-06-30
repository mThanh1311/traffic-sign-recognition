import cv2 as cv
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
from keras.models import load_model
from preprocessing import load_img, load_label_names, coverted_label

IMG_WIDTH = 30
IMG_HEIGHT = 30

#Load_model 
model = load_model(sys.argv[1])

#Load images
img_test = load_img(sys.argv[2])

pred = model.predict(img_test)
fig = plt.figure(figsize=(10,20))
for i, (image, prediction) in enumerate(zip(img_test, pred)):
    pred_classes = np.argmax(prediction)
    #load label names
    labels = coverted_label(pred_classes, sys.argv[3])
    plt.subplot(4,2,i+1)
    plt.imshow(image)
    plt.title(labels)
    plt.axis('off')
plt.show()