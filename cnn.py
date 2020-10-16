import cv2 as cv2
import numpy as np
import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train=x_train/255.0
x_test=x_test/255.0


#model11
model = tf.keras.Sequential([
   tf.keras.layers.AveragePooling2D(6,3, input_shape=(28,28,1)),
   tf.keras.layers.Conv2D(64, 3, activation='relu'),
   tf.keras.layers.Conv2D(32, 3, activation='relu'),
   tf.keras.layers.MaxPool2D(2,2),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)
###
#test_loss, test_acc = model.evaluate(x_test,  y_test) 
#print('Test accuracy:', test_acc)

model.save('model')
