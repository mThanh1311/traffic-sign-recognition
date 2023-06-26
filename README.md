# Traffic Sign Recognition
AI Project: Use CNN model and GTSRB for Traffic sign classification
# Overview
In this project, I used deep learning with Convolution Neural Network to classify the traffic sign. I will train and validate a model so it can classify traffic sign images using the German Traffic Sign Dataset. After the model is trained, I will then try out my model on images of Vietnamese traffic signs that I find on the web and I took a photo in the real-world.
# AI Model Overview
* Load and explore the data set
* Add the 2 Keras layers: Conv2D + MaxPooling2D (with pooling size is 2x2)
* Use ReLU for activation function in this two layers
* Make some modifications to reduce the Overfitting: Dropout, BatchNormaliztion
* Add some layers Dense for Hidden Layers
* Use adam optimization, categorical_crossentropy loss function
* Analyze the softmax probabilities of the new images
* Summarize the results (validation accuracy, accuracy, validation loss, loss)
# Dependencies
* Python: 3.10
* Tensorflow: 2.12.0
* Matplotlib: 3.7.0
* Numpy: 1.23.5
* OpenCV: 4.5.4.58
* scikit-learn: 1.2.2
# Dataset:
Download the [data set]([https://github.com](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)). This is a pickled dataset in which the images are already resized to 32x32. It contains a training, validation and test set.
# Training:
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

Run 'traffic.py' on Terminal:
'python traffic.py gtsrb [model.h5]'

The results are:
  - loss: 0.1573
  - accuracy: 0.9531 
  - validation loss: 0.0797 
  - valiadtion accuracy: 0.9794

We can see that the model is overfitting to the training data and the accuracy on validation set is a little lower than on training set

<img src="https://imgur.com/a/XpEV4Kn)https://imgur.com/a/XpEV4Kn">

# Output Test:
Run 'inference.py' on Terminal:
'python inference.py [model.h5] <test> <file.csv>'

<img src="(https://imgur.com/a/NkeomDm)https://imgur.com/a/NkeomDm">
