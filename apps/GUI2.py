#load the model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

import numpy as np


import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox


# input for efficientNet

import efficientnet.tfkeras as efn

from efficientnet.tfkeras import preprocess_input as effi_preprocess
from tensorflow.keras.applications.imagenet_utils import decode_predictions as effi_decode


#Creating the GUI
window = tk.Tk()
window.title("Image Recognition")



def loadImage():
    img_path = filedialog.askopenfilename()
    img = image.load_img(img_path, target_size=(224,224))
    global x
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    messagebox.showinfo('Success' ,'Load Picture Successfully!')


def VGG():
    pred = preprocess_input(x)
    model = VGG16(weights = 'imagenet', include_top=True)

    features = model.predict(pred)

    messagebox.showinfo('Predicted', 'This picture may be:\n '
                        +str(decode_predictions(features, top=3)[0][0][1])
                        +'\n Probobility:'+str(decode_predictions(features, top=3)[0][0][2]) 
                        )

def EfficientNet():
    effi = effi_preprocess(x)
    model = efn.EfficientNetB0(weights='imagenet')
    y = model.predict(effi)
    messagebox.showinfo('Predicted', 'This picture may be:\n '
                        +str(effi_decode(y, top=3)[0][0][1])
                        +'\n Probobility:'+str(effi_decode(y, top=3)[0][0][2]) 
                        )
    

button = tk.Button(window, text="Load Image", command=loadImage).grid(row=0,column=1)

button1 = tk.Button(window, text="VGG16 Prediction", command=VGG).grid(row=1, column=0)

button2 = tk.Button(window, text="EfficientNet Prediction", command=EfficientNet).grid(row=1, column=2)

window.mainloop()



