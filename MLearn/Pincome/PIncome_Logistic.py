import numpy as np
import keras
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense , Dropout, Activation
from keras.layers import Conv2D ,MaxPooling2D,Flatten
from keras.optimizers import  SGD,Adam
from keras.utils import np_utils

X_train_fpath = 'data/X_train'
Y_train_fpath = 'data/Y_train'
X_test_fpath = 'data/X_test'
output_fpath = 'output.csv'

X_train = np.genfromtxt(X_train_fpath, delimiter=',', skip_header=1)
Y_train = np.genfromtxt(Y_train_fpath, delimiter=',', skip_header=1)


def _normalize_column_normal(X, train=True, specified_column=None, X_mean=None, X_std=None):
    # The output of the function will make the specified column number to
    # become a Normal distribution
    # When processing testing data, we need to normalize by the value
    # we used for processing training, so we must save the mean value and
    # the variance of the training data
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column], 0), (1, length))
        X_std = np.reshape(np.std(X[:, specified_column], 0), (1, length))

    X[:, specified_column] = np.divide(np.subtract(X[:, specified_column], X_mean), X_std)

    return X


col = [0,1,3,4,5,7]
# scaler=StandardScaler()
# scaler.fit(X_train)
# X_train=scaler.transform(X_train)

X_train = _normalize_column_normal(X_train,specified_column=col)
#
# print(X_train[0])
Y_train=Y_train.reshape(32561,1)


model = Sequential()
model.add(Dense(input_dim=106,units=500,activation='sigmoid'))
model.add(Dense(units=500,activation='sigmoid'))
model.add(Dense(units=500,activation='sigmoid'))

model.add(Dense(units=1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer=Adam(0.001),metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=32,epochs=20)
X_test = np.genfromtxt(X_test_fpath, delimiter=',', skip_header=1)
X_test = _normalize_column_normal(X_test,specified_column=col)
# scaler.fit(X_test)
# X_test=scaler.transform(X_test)
result =np.round(model.predict(X_test))

with open('output.csv','w') as f:
    f.write('id,label\n')
    for i ,v in enumerate(result):
        f.write('%d ,%d\n' %(i+1,v))

