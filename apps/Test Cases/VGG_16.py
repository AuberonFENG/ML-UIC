#load the model


#load the file
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

img_path = filedialog.askopenfilename()


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights = 'imagenet', include_top=True)

img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

print('Predicted: ', decode_predictions(features, top=3)[0])




