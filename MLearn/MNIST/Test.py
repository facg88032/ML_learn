import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense , Dropout, Activation
from keras.layers import Conv2D ,MaxPooling2D,Flatten
from keras.optimizers import  SGD,Adam
from keras.utils import np_utils

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# 檢視結構
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print("---")


#print(np.argmax(y_train[1,:])) 查看label 用argmax轉化成數字
model = Sequential()
model.add(Dense(input_dim=28*28,units=500,activation='relu'))
model.add(Dense(units=500,activation='relu'))
model.add(Dense(units=500,activation='relu'))

model.add(Dense(units=10,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1),metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=100,epochs=5)


# result=model.evaluate(x_train,y_train) #check training data accuracy
#
# print('\nTrain Acc:',result[1])
#
# result=model.evaluate(x_test,y_test)
#
# print('\nTest Acc:',result[1])


result=model.predict(x_test)
print('real:',y_test[0])
print('predict:',result[0])

print('predict:',max(result[0]))