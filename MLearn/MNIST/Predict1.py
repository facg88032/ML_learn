import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense , Dropout, Activation
from keras.layers import Conv2D ,MaxPooling2D,Flatten
from keras.optimizers import  SGD,Adam
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model

test_model = load_model('MNIST.h5')
img = load_img('MNIST_data/five.png',False,target_size=(840,840))
print(img)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = test_model.predict_classes(x)
prob = test_model.predict_proba(x)
print(preds, prob)