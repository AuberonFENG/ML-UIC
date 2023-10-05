from PIL import Image

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

from efficientnet.tfkeras import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

model = efn.EfficientNetB0(
    weights='imagenet'
)

# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# features = model.predict(x)
# print(decode_predictions(features, top=3)[0])
# print(decode_predictions(features, top=3)[1])
# print(decode_predictions(features, top=3)[2])

# im = Image.open(img_path)
img_path= 'F:/ML/group/application/dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# make prediction and decode
y = model.predict(x)
print(decode_predictions(y)[0])
print("predicted class is %s, and predict probability is %f"%(decode_predictions(y)[0][0][1],decode_predictions(y)[0][0][2]))