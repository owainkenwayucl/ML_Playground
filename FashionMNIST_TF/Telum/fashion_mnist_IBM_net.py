# Based on the Tensorflow examples which have confusing licensing.

# Assume this is under the appropriate license (Apache and/or MIT), 
# converted from Jupyter Notebook by Owain Kenway

# This version has the more complicated network from the IBM example:
# https://github.com/IBM/ibmz-accelerated-for-tensorflow/blob/main/samples/fashion-mnist/fashion_mnist_training.py
# 
# Which is 
# Copyright 2023 IBM Corp, Licensed under the Apache License, Version 2.0 

# Original license statement is here:
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb
#
# Copyright 2018 The TensorFlow Authors. 
#
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import termshow

from termshow import ANSI_COLOURS

print(tf.__version__)

# Get the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Names of classes in the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
     
# Images are stored as integer greyscale - convert to float brightness
train_images = train_images / 255.0
test_images = test_images / 255.0

assert train_images.shape == (60000, 28, 28)

# This network is from the IBM example
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,7,padding='same',activation='relu', input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])


# Compile the model (from IBM example)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train it     
model.fit(train_images, train_labels, epochs=10)

# Test its loss on the test data aka how accurate is it
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
     
# Add a layer that shows us probabilities
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
# Display the first test image
termshow.show(test_images[0], colours=ANSI_COLOURS)

# Generate probabilities for the set of test images
predictions = probability_model.predict(test_images)

# Print the most likely class for the first image
print(class_names[np.argmax(predictions[0])])

