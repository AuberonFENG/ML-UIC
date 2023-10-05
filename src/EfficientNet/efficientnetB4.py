import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
# import tensorflow_datasets as tfds
# from tensorflow import keras
# from tensorflow.keras.callbacks import LearningRateScheduler


import os
import sys
import numpy as np
# from skimage.io import imread
import matplotlib.pyplot as plt


# below is test efficient net in keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras import optimizers
import efficientnet.tfkeras as efn

model = Sequential()

model.add(efn.EfficientNetB4(
    weights='imagenet',
    input_shape=(32,32,3),
    include_top=False,
    classes=10,
))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))  # classnumber 代表类别个数
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001),
                      metrics=['accuracy'])





def custom_accuracy(self, y_true, y_pred):
    predict = tf.reshape(y_pred, [-1, self.captcha_length, self.captcha_class])
    max_idx_p = tf.argmax(predict, 2)  # 这个做法牛逼，不用再做stack和reshape了，2，是在Charset那个维度上
    max_idx_l = tf.argmax(tf.reshape(y_true, [-1, self.captcha_length, self.captcha_class]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e), elems=correct_pred, dtype=tf.bool)
    return tf.reduce_mean(tf.cast(_result, tf.float32))

# cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1)
model.summary()

# 搞定，画图，显示训练集和验证集的acc和loss曲线
print(history.history.keys())

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
