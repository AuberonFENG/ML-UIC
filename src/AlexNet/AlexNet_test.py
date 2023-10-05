# Programmer: Ziao Feng Auberon 2030026036
# Task :
# Use TensorFlow to implement AlexNet

# we set batch size to 32, since the image data is large (224*224*3), large batch size will cause video access memory loss.
# here we don't use actual images to train, so num_batch can also use as the times of training.

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import math
import time
from datetime import datetime
batch_size = 32
num_batches = 100
# due to tensorflow version, w_init can use tf.truncated_normal() instead
# w_init = tf.random.truncated_normal(shape = shape, stddev=0.1,dtype = tf.float32)
b_init = tf.constant_initializer(0.1)

# show the version of tensorflow and keras
print("Keras Version: ",keras.__version__)
print("TensorFlow Version: ",tf.__version__)

# we don't have API in keras to implement LRN.
# so we use function nn.lrn() to implement LRN function
# thus we need to define subclass of Model and implement each conv layer.
# nn model: refer to 
# https://tendorflow.google.cn/versions/r2.0/api_docs/python/tf/nn

# AlexNet use ReLU as activation function instead of Sigmoid to prevent the gradient vanishing problem that happens in deep layer.

# AlexNet use a LRN layer.
# For the use of LRN layer, please refer to the essay: ImageNet Classification with Deep Convolutional Neural Networks

# Here we use parameters recomanded in that essay.
# depth_radius=4, bias=1, alpha=0.001/9, beta=0.75

# remind: LRN is more useful to the activation function that don't have upper limit
# that's because LRN choose the larger Response from those Conv Kernals. 
# Functions like Sigmoid and tanh will limit the extra big input value, since the value won't have much variations when input and output value is huge.

# AlexNet use maxpooling instead of average pooling.

# Next is the first covlution layer: Conv1
class Conv1 (layers.Layer):
    def __init__(self):
        super(Conv1, self).__init__()
        
    def build(self, input_shape):
        self.kernel = tf.Variable(tf.random.truncated_normal([11,11,3,96],stddev=0.1,dtype = tf.float32),name='Conv1/kernel')  
            # add_weight(name='Conv1/kernel', shape=[11,11,3,96],initializer=tf.random.truncated_normal([11,11,3,96],stddev=0.1,dtype = tf.float32),dtype='float32',trainable=True)
        self.biases = self.\
            add_weight(name='Conv1/biases',shape=[96],initializer=b_init,dtype='float32',trainable=True)
    def call(self, inputs):
        conv = tf.nn.conv2d(inputs, self.kernel, [1, 4, 4, 1], padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.biases))
        lrn = tf.nn.lrn(relu,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name="Conv1/lrn")
        pool = tf.nn.max_pool(lrn, ksize=[1, 3, 3, 1],strides=[1,2,2,1],padding="VALID", name="Conv1/pool")
        return pool
        
# next is the second conv layer, 5*5 kernal, input channel 96, stride 1, convkernal number 256
class Conv2 (layers.Layer):
    def __init__(self):
        super(Conv2, self).__init__()
        
    def build(self, input_shape):
        self.kernel = tf.Variable(tf.random.truncated_normal([5,5,96,256],stddev=0.1,dtype = tf.float32),name='Conv2/kernel')
            # add_weight(name='Conv2/kernel', shape=[5,5,96,256],initializer=tf.random.truncated_normal([5,5,96,256],stddev=0.1,dtype = tf.float32),dtype='float32',trainable=True)
        self.biases = self.\
            add_weight(name='Conv2/biases',shape=[256],initializer=b_init,dtype='float32',trainable=True)
    def call(self, inputs):
        conv = tf.nn.conv2d(inputs, self.kernel, [1, 1, 1, 1], padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.biases))
        lrn = tf.nn.lrn(relu,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name="Conv2/lrn")
        pool = tf.nn.max_pool(lrn, ksize=[1, 3, 3, 1],strides=[1,2,2,1],padding="VALID", name="Conv2/pool")
        return pool

# third layer Conv layer 3
# 3*3 kernel, 256 channels, num of kernel 384, stride 1
# no pooling and LRN

class Conv3 (layers.Layer):
    def __init__(self):
        super(Conv3, self).__init__()
        
    def build(self, input_shape):
        self.kernel = tf.Variable(tf.random.truncated_normal([3,3,256,384],stddev=0.1,dtype = tf.float32),name='Conv3/kernel')
            # add_weight(name='Conv3/kernel', shape=[3,3,256,384],initializer=tf.random.truncated_normal([3,3,256,384],stddev=0.1,dtype = tf.float32),dtype='float32',trainable=True)
        self.biases = self.\
            add_weight(name='Conv3/biases',shape=[384],initializer=b_init,dtype='float32',trainable=True)
    def call(self, inputs):
        conv = tf.nn.conv2d(inputs, self.kernel, [1, 1, 1, 1], padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.biases))
        return relu

# fourth conv layer 4
# 3*3 kernel, 384 channels, num of kernel 384, stride 1
# no pooling and LRN

class Conv4 (layers.Layer):
    def __init__(self):
        super(Conv4, self).__init__()
        
    def build(self, input_shape):
        self.kernel = tf.Variable(tf.random.truncated_normal([3,3,384,384],stddev=0.1,dtype = tf.float32),name='Conv4/kernel')
            # add_weight(name='Conv4/kernel', shape=[3,3,384,384],initializer=tf.random.truncated_normal([3,3,384,384],stddev=0.1,dtype = tf.float32),dtype='float32',trainable=True)
        self.biases = self.\
            add_weight(name='Conv4/biases',shape=[384],initializer=b_init,dtype='float32',trainable=True)
    def call(self, inputs):
        conv = tf.nn.conv2d(inputs, self.kernel, [1, 1, 1, 1], padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.biases))
        return relu

# fifth conv layer 5
# kernel 3*3, channel 384, num of kernal 256, stride 1
# use maxpooling, but no lrn


class Conv5 (layers.Layer):
    def __init__(self):
        super(Conv5, self).__init__()
        
    def build(self, input_shape):
        self.kernel = tf.Variable(tf.random.truncated_normal([3,3,384,256],stddev=0.1,dtype = tf.float32),name='Conv5/kernel')
            # add_weight(name='Conv5/kernel', shape=[3,3,384,256],initializer=tf.random.truncated_normal([3,3,384,256],stddev=0.1,dtype = tf.float32),dtype='float32',trainable=True)
        self.biases = self.\
            add_weight(name='Conv5/biases',shape=[256],initializer=b_init,dtype='float32',trainable=True)
    def call(self, inputs):
        conv = tf.nn.conv2d(inputs, self.kernel, [1, 1, 1, 1], padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1],strides=[1,2,2,1],padding="VALID", name="Conv5/pool")
        return pool

# next use flatten to reduce dimension
# add three dense layer 
# here we only have one GPU, so the first two dense layer have 4096 hidden units
# the third dense layer use softmax to classify 1000 hidden units

# after each dense , we need to dropout, here it doea not influence the time of backpropagation and forward propagation, but to keep the model perfect, we still add dropout here.

class AlexNet (tf.keras.Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = Conv1()
        self.conv2 = Conv2()
        self.conv3 = Conv3()
        self.conv4 = Conv4()
        self.conv5 = Conv5()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=4096, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(units=4096, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.dense3 = layers.Dense(units=1000, activation='softmax')
    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.dense3(x)
    

# we create a test dataset to run this AlexNet

image_size = 224
image_shape = [batch_size, image_size, image_size, 3]
image_init = tf.random_normal_initializer(stddev=1e-1)
image_data = tf.Variable(initial_value=image_init(shape=image_shape),dtype='float32')

# print the model info

alexnet = AlexNet()
alexnet(image_data)
alexnet.summary()

# next try to evaluate AlexNet on forward propagation
total_dura = 0.0
total_dura_squared = 0.0

for step in range(num_batches+10):
    start_time = time.time()
    alexnet(image_data)
    duration = time.time() - start_time
    if step >= 10:
        if step%10==0:
            print('%s: step %d, duration = %.3f'% (datetime.now(), step-10, duration))
        total_dura += duration
        total_dura_squared += duration + duration
average_time = total_dura / num_batches

#print forward propagation info

print('%s:Foward across %d steps,%.3f +/- %.3f sec/batch'% (datetime.now(), num_batches, average_time, math.sqrt(total_dura_squared/num_batches-average_time*average_time)))

'''
# try to show time on backward propagation
back_total_dura = 0.0
back_total_dura_squared = 0.0

for step in range(num_batches + 10):
    start_time = time.time()
    with tf.GradientTape() as tape:
        loss = tf.nn.l2_loss(alexnet(image_data))
    gradients = tape.gradient(loss,alexnet.trainable_variables)

duration = time.time

'''