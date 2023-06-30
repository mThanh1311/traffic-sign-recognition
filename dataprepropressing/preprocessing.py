import csv 
import cv2 as cv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

IMG_WIDTH = 30
IMG_HEIGHT = 30

#Load image for tesing
def load_img(img_path):
    #List for image
    images_for_test = []

    for img_file in glob.glob(img_path+'/*.jpg'):
        img = cv.imread(img_file)
        img = cv.resize(img, (IMG_WIDTH,IMG_HEIGHT), interpolation= cv.INTER_AREA)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  
        img_converted = img_rgb / 255.0
        images_for_test.append(img_converted)
    images_for_test = np.array(images_for_test) 

    return (images_for_test)

#Load label name from file.csv
def load_label_names(csv_path):
    label_names = {}
    with open(csv_path) as file:
        reader = csv.reader(file)
        #bo qua phan tieu de cua file.csv
        next (reader)
        for row in reader:
            #Add key-value into label_name
            label_names[int(row[0])] = row[1]
    return label_names

#Get a VALUE from label_names dictionary
def coverted_label(index, csv_path):
    label = load_label_names(csv_path)
    label_name = label[index]
    return label_name