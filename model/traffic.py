import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import utils
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    images = np.array(images)
    images = images/255
    labels = utils.to_categorical(labels) #one-hot-coding
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images),
        np.array(labels),
        test_size=TEST_SIZE
    )

    #Get a compiled neural network
    model = get_model()

    #Get a summary layers of model
    model.summary()

    #Fit model on training data
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

    #Draw graphs accuracy and loss
    drw = draw_graph(history)


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    # Path to data folder
    data_path = os.path.join(data_dir)

    # Number of subdirectories/labels
    number_of_labels = 0
    
    for i in os.listdir(data_path):
        number_of_labels += 1

    # Loop through the subdirectories
    for sub in range(number_of_labels):
        sub_folder = os.path.join(data_path, str(sub))

        images_in_subfolder = []

        for image in os.listdir(sub_folder):
            images_in_subfolder.append(image)

        # Open each image 
        for image in images_in_subfolder:

            image_path = os.path.join(data_path, str(sub), image)

            # Add Label
            labels.append(sub)

            # Resize and Add Image
            img = cv.imread(image_path)

            # print(image_path)
            res = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation= cv.INTER_AREA)
            images.append(res)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = models.Sequential([

        # 2 Convolutional layers and 2 Max-pooling layers
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        layers.BatchNormalization(),

        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Dropout(0,2),

        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        layers.BatchNormalization(),

        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Dropout(0,2),

        # Flatten units
        layers.Flatten(),

        # Hidden Layers
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),

        # Dropout
        layers.Dropout(0.3),

        # Extra hidden layer
        layers.Dense(48, activation="relu"),

        # Output layer with output units for all digits
        layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def draw_graph(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
    main()

