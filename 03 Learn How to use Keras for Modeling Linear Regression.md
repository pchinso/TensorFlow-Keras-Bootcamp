# Learn How to use Keras for Modeling Linear Regression

Before
 studying deep neural networks, we will cover the fundamental components
 of a simple (linear) neural network. We'll begin with the topic of
linear regression. Since linear regression can be modeled as a neural
network, it provides an excellent example to introduce the essential
components of neural networks. Regression is a form of supervised
learning which aims to model the relationship between one or more input
variables (features) and a continuous (target) variable. We assume that
the relationship between the input variables **x**

 and the target variable **y**
 can be expressed as a weighted sum of the inputs (i.e., the model is
linear in the parameters). In short, linear regression aims to learn a
function that maps one or more input features to a single numerical
target value.

![](https://learnopencv.com/wp-content/uploads/2023/01/keras-linear-regression-model-plot.png)

## Table of Contents

* [1 Dataset Exploration](https://courses.opencv.org/asset-v1:Tensorflow+Bootcamp+TFKS+type@asset+block/03-Keras-Linear-Regression.html#1-Dataset-Exploration)
* [2 Linear Regression Model](https://courses.opencv.org/asset-v1:Tensorflow+Bootcamp+TFKS+type@asset+block/03-Keras-Linear-Regression.html#2-Linear-Regression-Model)
* [3 Neural Network Perspective and Terminology](https://courses.opencv.org/asset-v1:Tensorflow+Bootcamp+TFKS+type@asset+block/03-Keras-Linear-Regression.html#3-Neural-Network-Perspective-and-Terminology)
* [4 Modeling a Neural Network in Keras](https://courses.opencv.org/asset-v1:Tensorflow+Bootcamp+TFKS+type@asset+block/03-Keras-Linear-Regression.html#4-Modeling-a-Neural-Network-in-Keras)
* [5 Conclusion](https://courses.opencv.org/asset-v1:Tensorflow+Bootcamp+TFKS+type@asset+block/03-Keras-Linear-Regression.html#5-Conclusion)

```
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import layers

import tensorflow as tf
import matplotlib.pyplot as plt
```

```
SEED_VALUE = 42

# Fix seed to make training deterministic.
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
```

# 1 Dataset Exploration

## 1.1 Load the Boston Housing Dataset

In
 this notebook, we will be working with the Boston Housing dataset. This
 dataset contains information collected by the U.S Census Service
concerning housing in Boston MA. It has been used extensively throughout
 the literature to benchmark algorithms and is also suitable for
demonstration purposes due to its small size. The dataset contains 14
unique attributes, among which is the median value (price in $K) of a
home for a given suburb. We will use this dataset as an example of how
to develop a model that allows us to predict the median price of a home
based on a single attribute in the dataset (average number of rooms in a
 house).

Keras provides the `load_data()` function to load this dataset. Datasets are typically partitioned into `train`, and `test` components, and the `load_data()`
 function returns a tuple for each. Each tuple contains a 2-dimensional
array of features (e.g., X_train) and a vector that contains the
associated target values for each sample in the dataset (e.g., y_train).
 So, for example, the rows in `X_train` represent the various
 samples in the dataset and the columns represent the various features.
In this notebook, we are only going to make use of the training data to
demonstrate how to train a model. However, in practice, it is very
important to use the test data to see how well the trained model
performs on unseen data.

```
# Load the Boston housing dataset.
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

print(X_train.shape)
print("\n")
print("Input features: ", X_train[0])
print("\n")
print("Output target: ", y_train[0])
```

```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz
57026/57026 [==============================] - 0s 0us/step
(404, 13)


Input features:  [  1.23247   0.        8.14      0.        0.538     6.142    91.7
   3.9769    4.      307.       21.      396.9      18.72   ]


Output target:  15.2
```

## 1.2 Extract Features from the Dataset

In
 this notebook we are only going to use a single feature from the
dataset, so to keep things simple, we will store the feature data in a
new variable.

```
boston_features = {
    "Average Number of Rooms": 5,
}

X_train_1d = X_train[:, boston_features["Average Number of Rooms"]]
print(X_train_1d.shape)

X_test_1d = X_test[:, boston_features["Average Number of Rooms"]]
```

```
(404,)
```

## 1.3 Plot the Features

Here we plot the median price of a home vs. the single feature ('Average Number of Rooms').

```
plt.figure(figsize=(15, 5))

plt.xlabel("Average Number of Rooms")
plt.ylabel("Median Price [$K]")
plt.grid("on")
plt.scatter(X_train_1d[:], y_train, color="green", alpha=0.5)
```

```
<matplotlib.collections.PathCollection at 0x7f21bbde9d60>
```

![]()

## 2 Linear Regression Model

Let's
 first start with a clear picture of what we are trying to accomplish.
The plot below shows the training data for the single independent
variable (number of rooms) and the dependent variable (the median price
of a house). We would like to use linear regression to develop a
reliable model for this data. In this example, the model is simply a
straight line defined by its slope (**m**

) and y-intercept (**b**).

![](https://learnopencv.com/wp-content/uploads/2023/01/keras-linear-regression-model-plot.png)

## 3 Neural Network Perspective and Terminology

The
 figure below shows how this model can be represented as a simple
(single neuron) network. We will use this simple example to introduce
neural network components and terminology. The input data (**x**

) consists of a single feature (average number of rooms), and the predicted output (**y**′)
 is a scalar (predicted median price of a home). Note that each data
sample in the dataset represents the statistics for a Boston suburb. The
 model parameters (**m** and **b**),
 are learned iteratively during the training process. As you may already
 know, the model parameters can be computed by the method of Ordinary
Least Squares (OSL) in the closed form. However, we can also solve this
problem iteratively using a numerical technique called  **Gradient Descent** ,
 which is the basis for how neural networks are trained. We will not
cover the details of gradient descent in this notebook, but it's
important to understand that it's an iterative technique that is used to
 tune the parameters of the model.

![](https://learnopencv.com/wp-content/uploads/2023/01/keras-linear-regression-forward-pass-block-diagram.png)

The network contains just a single neuron that takes a single input (**x**

) and produces a single output (**y**′) which is the predicted (average) price of a home. The single neuron has two trainable parameters, which are the slope (**m**) and y-intercept (**b**)
 of the linear model. These parameters are more generally known as the
weight and bias, respectively. In regression problems, it is common for
the model to have multiple input features, where each input has an
associated weight (**w**i),
 but in this example, we will use just a single input feature to predict
 the output. So, in general, a neuron typically has multiple weights (**w**1, **w**2, **w**3, etc.) and a single bias term (**b**). In this example, you can think of the neuron as the mathematical computation of **m**x**+**b, which produces the predicted value **y**′.

A slightly more formal diagram is shown below for the same model. Here we have now introduced the concept of a **feedback loop** that shows how model parameters (**w**

 and **b**)
 are updated during the training process. Initially, the model
parameters are initialized to small random values. During the training
process, as training data is passed through the network, the predicted
value of the model (**y**′) is compared to the ground truth (**y**) for a given sample from the dataset. The difference is used as the basis to compute a **loss**
 which is then used as feedback in the network to adjust the model
parameters in a way that improves the prediction. This process involves
two steps called **Gradient Descent** and  **Backpropagation** .
 It's not important at this stage to understand the mathematical details
 of how this works, but it is important to understand that there is an
iterative process to training the model.

![](https://learnopencv.com/wp-content/uploads/2023/01/keras-linear-regression-weight-update-block-diagram.png)

The **Loss Function** we use can take many forms. For this example, we will use **Mean Squared Error (MSE)** which is a very common loss function used in regression problems.

**J**=**1**m**m**∑**i**=**1**(**y**′**i**−**y**i**)**2The
 basic idea is that we want to minimize the value of this function which
 is a representation of the error between our model and the training
dataset. In the equation above, **m**

 is the number of training samples.

## 4 Modeling a Neural Network in Keras

The
 network diagram in the previous section represents the simplest
possible neural network. The network has a single layer consisting of a
single neuron that outputs **w**x**+**b

. For every training sample, the predicted output **y**′
 is compared to the actual value from the training data, and the loss is
 computed. The loss can then be used to fine-tune (update) the model
parameters.

All of the details associated with training a neural network are taken care of by Keras as summarized in the following workflow:

1. Build/Define a network model using predefined layers in Keras.
2. Compile the model with `model.compile()`
3. Train the model with `model.fit()`
4. Predict the output `model.predict()`

### 4.1 Define the Keras Model

```
model = Sequential()

# Define the model consisting of a single neuron.
model.add(Dense(units=1, input_shape=(1,)))

# Display a summary of the model architecture.
model.summary()
```

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 1)                 2       
                                                               
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________
```

### 4.2 Compile the Model

```
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005), loss="mse")
```

### 4.3 Train the Model

```
history = model.fit(
    X_train_1d, 
    y_train, 
    batch_size=16, 
    epochs=101, 
    validation_split=0.3,
)
```

```
Epoch 1/101
18/18 [==============================] - 1s 18ms/step - loss: 218.3039 - val_loss: 266.6791
Epoch 2/101
18/18 [==============================] - 0s 5ms/step - loss: 200.4328 - val_loss: 248.9743
Epoch 3/101
18/18 [==============================] - 0s 6ms/step - loss: 185.4961 - val_loss: 232.8605
Epoch 4/101
18/18 [==============================] - 0s 6ms/step - loss: 171.7416 - val_loss: 217.5213
Epoch 5/101
18/18 [==============================] - 0s 5ms/step - loss: 158.4702 - val_loss: 202.5981
Epoch 6/101
18/18 [==============================] - 0s 4ms/step - loss: 146.0787 - val_loss: 188.8977
Epoch 7/101
18/18 [==============================] - 0s 4ms/step - loss: 134.7131 - val_loss: 175.8734
Epoch 8/101
18/18 [==============================] - 0s 3ms/step - loss: 124.2067 - val_loss: 164.1291
Epoch 9/101
18/18 [==============================] - 0s 4ms/step - loss: 114.4688 - val_loss: 152.7321
Epoch 10/101
18/18 [==============================] - 0s 3ms/step - loss: 105.3791 - val_loss: 142.2674
Epoch 11/101
18/18 [==============================] - 0s 8ms/step - loss: 97.3249 - val_loss: 132.7749
Epoch 12/101
18/18 [==============================] - 0s 5ms/step - loss: 89.7307 - val_loss: 123.7139
Epoch 13/101
18/18 [==============================] - 0s 6ms/step - loss: 83.1570 - val_loss: 115.9223
Epoch 14/101
18/18 [==============================] - 0s 4ms/step - loss: 77.4052 - val_loss: 108.6153
Epoch 15/101
18/18 [==============================] - 0s 4ms/step - loss: 72.3634 - val_loss: 102.4045
Epoch 16/101
18/18 [==============================] - 0s 4ms/step - loss: 68.2201 - val_loss: 97.1872
Epoch 17/101
18/18 [==============================] - 0s 4ms/step - loss: 64.6780 - val_loss: 92.2217
Epoch 18/101
18/18 [==============================] - 0s 3ms/step - loss: 61.7445 - val_loss: 88.3425
Epoch 19/101
18/18 [==============================] - 0s 3ms/step - loss: 59.6233 - val_loss: 85.0607
Epoch 20/101
18/18 [==============================] - 0s 4ms/step - loss: 57.8111 - val_loss: 82.0434
Epoch 21/101
18/18 [==============================] - 0s 4ms/step - loss: 56.5885 - val_loss: 80.2545
Epoch 22/101
18/18 [==============================] - 0s 3ms/step - loss: 55.8929 - val_loss: 78.7788
Epoch 23/101
18/18 [==============================] - 0s 4ms/step - loss: 55.3621 - val_loss: 77.5162
Epoch 24/101
18/18 [==============================] - 0s 3ms/step - loss: 55.0594 - val_loss: 76.9467
Epoch 25/101
18/18 [==============================] - 0s 3ms/step - loss: 54.9246 - val_loss: 76.5705
Epoch 26/101
18/18 [==============================] - 0s 3ms/step - loss: 54.8514 - val_loss: 76.1284
Epoch 27/101
18/18 [==============================] - 0s 3ms/step - loss: 54.7593 - val_loss: 75.8563
Epoch 28/101
18/18 [==============================] - 0s 4ms/step - loss: 54.7621 - val_loss: 75.7362
Epoch 29/101
18/18 [==============================] - 0s 4ms/step - loss: 54.7170 - val_loss: 75.4796
Epoch 30/101
18/18 [==============================] - 0s 3ms/step - loss: 54.7375 - val_loss: 75.4277
Epoch 31/101
18/18 [==============================] - 0s 3ms/step - loss: 54.7118 - val_loss: 75.3171
Epoch 32/101
18/18 [==============================] - 0s 3ms/step - loss: 54.6758 - val_loss: 75.3104
Epoch 33/101
18/18 [==============================] - 0s 4ms/step - loss: 54.6451 - val_loss: 75.2866
Epoch 34/101
18/18 [==============================] - 0s 3ms/step - loss: 54.6635 - val_loss: 75.1464
Epoch 35/101
18/18 [==============================] - 0s 3ms/step - loss: 54.6305 - val_loss: 74.9947
Epoch 36/101
18/18 [==============================] - 0s 4ms/step - loss: 54.6263 - val_loss: 74.9861
Epoch 37/101
18/18 [==============================] - 0s 3ms/step - loss: 54.6305 - val_loss: 74.9837
Epoch 38/101
18/18 [==============================] - 0s 4ms/step - loss: 54.5833 - val_loss: 74.8805
Epoch 39/101
18/18 [==============================] - 0s 4ms/step - loss: 54.5594 - val_loss: 74.9123
Epoch 40/101
18/18 [==============================] - 0s 3ms/step - loss: 54.6198 - val_loss: 74.9566
Epoch 41/101
18/18 [==============================] - 0s 3ms/step - loss: 54.5298 - val_loss: 75.0215
Epoch 42/101
18/18 [==============================] - 0s 4ms/step - loss: 54.5540 - val_loss: 74.9482
Epoch 43/101
18/18 [==============================] - 0s 4ms/step - loss: 54.5689 - val_loss: 74.8092
Epoch 44/101
18/18 [==============================] - 0s 3ms/step - loss: 54.5321 - val_loss: 74.9181
Epoch 45/101
18/18 [==============================] - 0s 3ms/step - loss: 54.5144 - val_loss: 74.8582
Epoch 46/101
18/18 [==============================] - 0s 3ms/step - loss: 54.5020 - val_loss: 74.7971
Epoch 47/101
18/18 [==============================] - 0s 5ms/step - loss: 54.4932 - val_loss: 74.8450
Epoch 48/101
18/18 [==============================] - 0s 3ms/step - loss: 54.5065 - val_loss: 74.8663
Epoch 49/101
18/18 [==============================] - 0s 3ms/step - loss: 54.4649 - val_loss: 74.7841
Epoch 50/101
18/18 [==============================] - 0s 4ms/step - loss: 54.4808 - val_loss: 74.8178
Epoch 51/101
18/18 [==============================] - 0s 4ms/step - loss: 54.4500 - val_loss: 74.7865
Epoch 52/101
18/18 [==============================] - 0s 4ms/step - loss: 54.4415 - val_loss: 74.8419
Epoch 53/101
18/18 [==============================] - 0s 3ms/step - loss: 54.4109 - val_loss: 74.7314
Epoch 54/101
18/18 [==============================] - 0s 4ms/step - loss: 54.4310 - val_loss: 74.7621
Epoch 55/101
18/18 [==============================] - 0s 4ms/step - loss: 54.3817 - val_loss: 74.8634
Epoch 56/101
18/18 [==============================] - 0s 4ms/step - loss: 54.3621 - val_loss: 74.8880
Epoch 57/101
18/18 [==============================] - 0s 3ms/step - loss: 54.4146 - val_loss: 74.7993
Epoch 58/101
18/18 [==============================] - 0s 3ms/step - loss: 54.3921 - val_loss: 74.8146
Epoch 59/101
18/18 [==============================] - 0s 3ms/step - loss: 54.3542 - val_loss: 74.7960
Epoch 60/101
18/18 [==============================] - 0s 3ms/step - loss: 54.3436 - val_loss: 74.5772
Epoch 61/101
18/18 [==============================] - 0s 3ms/step - loss: 54.3186 - val_loss: 74.6245
Epoch 62/101
18/18 [==============================] - 0s 3ms/step - loss: 54.3108 - val_loss: 74.6517
Epoch 63/101
18/18 [==============================] - 0s 3ms/step - loss: 54.3029 - val_loss: 74.7150
Epoch 64/101
18/18 [==============================] - 0s 3ms/step - loss: 54.3088 - val_loss: 74.7468
Epoch 65/101
18/18 [==============================] - 0s 4ms/step - loss: 54.2562 - val_loss: 74.5794
Epoch 66/101
18/18 [==============================] - 0s 5ms/step - loss: 54.2897 - val_loss: 74.4839
Epoch 67/101
18/18 [==============================] - 0s 4ms/step - loss: 54.2852 - val_loss: 74.4299
Epoch 68/101
18/18 [==============================] - 0s 3ms/step - loss: 54.2472 - val_loss: 74.5234
Epoch 69/101
18/18 [==============================] - 0s 3ms/step - loss: 54.2410 - val_loss: 74.5546
Epoch 70/101
18/18 [==============================] - 0s 4ms/step - loss: 54.2123 - val_loss: 74.7001
Epoch 71/101
18/18 [==============================] - 0s 4ms/step - loss: 54.2234 - val_loss: 74.5940
Epoch 72/101
18/18 [==============================] - 0s 4ms/step - loss: 54.2075 - val_loss: 74.5663
Epoch 73/101
18/18 [==============================] - 0s 4ms/step - loss: 54.1709 - val_loss: 74.3834
Epoch 74/101
18/18 [==============================] - 0s 4ms/step - loss: 54.1790 - val_loss: 74.5317
Epoch 75/101
18/18 [==============================] - 0s 4ms/step - loss: 54.1853 - val_loss: 74.4755
Epoch 76/101
18/18 [==============================] - 0s 4ms/step - loss: 54.1872 - val_loss: 74.5117
Epoch 77/101
18/18 [==============================] - 0s 3ms/step - loss: 54.1313 - val_loss: 74.4833
Epoch 78/101
18/18 [==============================] - 0s 4ms/step - loss: 54.1677 - val_loss: 74.5130
Epoch 79/101
18/18 [==============================] - 0s 4ms/step - loss: 54.1580 - val_loss: 74.4637
Epoch 80/101
18/18 [==============================] - 0s 4ms/step - loss: 54.1109 - val_loss: 74.3273
Epoch 81/101
18/18 [==============================] - 0s 4ms/step - loss: 54.0919 - val_loss: 74.4436
Epoch 82/101
18/18 [==============================] - 0s 4ms/step - loss: 54.1316 - val_loss: 74.4499
Epoch 83/101
18/18 [==============================] - 0s 3ms/step - loss: 54.0931 - val_loss: 74.4721
Epoch 84/101
18/18 [==============================] - 0s 4ms/step - loss: 54.0879 - val_loss: 74.4184
Epoch 85/101
18/18 [==============================] - 0s 3ms/step - loss: 54.0904 - val_loss: 74.2968
Epoch 86/101
18/18 [==============================] - 0s 4ms/step - loss: 54.0727 - val_loss: 74.3065
Epoch 87/101
18/18 [==============================] - 0s 4ms/step - loss: 54.0530 - val_loss: 74.2890
Epoch 88/101
18/18 [==============================] - 0s 4ms/step - loss: 54.0470 - val_loss: 74.3096
Epoch 89/101
18/18 [==============================] - 0s 3ms/step - loss: 54.0411 - val_loss: 74.2401
Epoch 90/101
18/18 [==============================] - 0s 3ms/step - loss: 54.0291 - val_loss: 74.3038
Epoch 91/101
18/18 [==============================] - 0s 3ms/step - loss: 53.9924 - val_loss: 74.3393
Epoch 92/101
18/18 [==============================] - 0s 4ms/step - loss: 53.9872 - val_loss: 74.2191
Epoch 93/101
18/18 [==============================] - 0s 4ms/step - loss: 54.0073 - val_loss: 74.1595
Epoch 94/101
18/18 [==============================] - 0s 3ms/step - loss: 53.9746 - val_loss: 74.2865
Epoch 95/101
18/18 [==============================] - 0s 4ms/step - loss: 53.9726 - val_loss: 74.1508
Epoch 96/101
18/18 [==============================] - 0s 4ms/step - loss: 53.9591 - val_loss: 74.0884
Epoch 97/101
18/18 [==============================] - 0s 3ms/step - loss: 53.9428 - val_loss: 74.1000
Epoch 98/101
18/18 [==============================] - 0s 4ms/step - loss: 53.9320 - val_loss: 74.1424
Epoch 99/101
18/18 [==============================] - 0s 3ms/step - loss: 53.9078 - val_loss: 74.1654
Epoch 100/101
18/18 [==============================] - 0s 3ms/step - loss: 53.9164 - val_loss: 74.0994
Epoch 101/101
18/18 [==============================] - 0s 3ms/step - loss: 53.8947 - val_loss: 74.0549
```

### 4.4 Plot the Training Results

```
def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'], 'g', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlim([0, 100])
    plt.ylim([0, 300])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
```

```
plot_loss(history)
```

![]()

The loss curves above are fairly typical. First, notice that there
are two curves, one for the training loss and one for the validation
loss. Both are large initially and then steadily decrease and eventually
 level off with no further improvement after about 30 epochs. Since the
model is only trained on the training data, it is also fairly typical
that the training loss is lower than the validation loss.

### 4.4 Make Predictions using the Model

We can now use the `predict()`
 method in Keras to make a single prediction. In this example, we pass a
 list of values to the model (representing the average number of rooms),
 and the model returns the predicted value for the price of a home for
each input.

```
# Predict the median price of a home with [3, 4, 5, 6, 7] rooms.
x = [3, 4, 5, 6, 7]
y_pred = model.predict(x)
for idx in range(len(x)):
    print(f"Predicted price of a home with {x[idx]} rooms: ${int(y_pred[idx] * 10) / 10}K")
```

```
1/1 [==============================] - 0s 99ms/step
Predicted price of a home with 3 rooms: $11.0K
Predicted price of a home with 4 rooms: $14.4K
Predicted price of a home with 5 rooms: $17.9K
Predicted price of a home with 6 rooms: $21.3K
Predicted price of a home with 7 rooms: $24.8K
```

### 4.5 Plot the Model and the Data

```
# Generate feature data that spans the range of interest for the independent variable.
x = np.linspace(3, 9, 10)

# Use the model to predict the dependent variable.
y = model.predict(x)
```

```
1/1 [==============================] - 0s 44ms/step
```

```
def plot_data(x_data, y_data, x, y, title=None):
  
    plt.figure(figsize=(15,5))
    plt.scatter(x_data, y_data, label='Ground Truth', color='green', alpha=0.5)
    plt.plot(x, y, color='k', label='Model Predictions')
    plt.xlim([3,9])
    plt.ylim([0,60])
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Price [$K]')
    plt.title(title)
    plt.grid(True)
    plt.legend()
```

```
plot_data(X_train_1d, y_train, x, y, title='Training Dataset')
```

![]()

```
plot_data(X_test_1d, y_test, x, y, title='Test Dataset')
```

![]()

## 5 Conclusion

In this notebok, we introduced the topic of linear regression in the
context of a simple neural network. We showed how Keras can be used to
model and train the network to learn the parameters of the linear model
and how to visualize the model predictions.
