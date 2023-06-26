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
# Summary Model:
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        896       

 batch_normalization (BatchN  (None, 28, 28, 32)       128       
 ormalization)

 max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0
 )

 dropout (Dropout)           (None, 14, 14, 32)        0

 conv2d_1 (Conv2D)           (None, 12, 12, 32)        9248

 batch_normalization_1 (Batc  (None, 12, 12, 32)       128
 hNormalization)

 max_pooling2d_1 (MaxPooling  (None, 6, 6, 32)         0
 2D)

 dropout_1 (Dropout)         (None, 6, 6, 32)          0

 flatten (Flatten)           (None, 1152)              0

 dense (Dense)               (None, 128)               147584

 dropout_2 (Dropout)         (None, 128)               0

 dense_1 (Dense)             (None, 128)               16512

 dropout_3 (Dropout)         (None, 128)               0

 dense_2 (Dense)             (None, 128)               16512

 dropout_4 (Dropout)         (None, 128)               0

 dense_3 (Dense)             (None, 48)                6192

 dense_4 (Dense)             (None, 43)                2107

=================================================================
Total params: 199,307

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

![z4466130561739_d750317ddea9e830b0c5fb200b30ef76](https://github.com/mThanh1311/traffic-sign-recognition/assets/89265290/13759ee6-8731-4480-b4a7-b66b5bf42dcf)
