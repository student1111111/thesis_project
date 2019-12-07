from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np

m = Sequential()
m.add(Dense(units=512,activation = 'relu', input_dim = 21))
m.add(Dropout(0.5))
m.add(Dense(units=512,activation = 'relu'))
m.add(Dropout(0.5))
m.add(Dense(units=512,activation = 'relu', input_dim = 21))
m.add(Dropout(0.5))
m.add(Dense(units=512,activation = 'relu', input_dim = 21))
m.add(Dropout(0.5))
m.add(Dense(units=512,activation = 'relu', input_dim = 21))
m.add(Dropout(0.5))
m.add(Dense(units=512,activation = 'relu', input_dim = 21))
m.add(Dropout(0.5))
m.add(Dense(units=512,activation = 'relu', input_dim = 21))
m.add(Dropout(0.5))
m.add(Dense(units=512,activation = 'relu', input_dim = 21))
m.add(Dropout(0.5))
m.add(Dense(units=512,activation = 'relu', input_dim = 21))
m.add(Dropout(0.5))
m.add(Dense(units=10,activation = 'softmax'))
m.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])



import pandas as pd

test_df=pd.read_csv("Desktop/test.csv",delimiter=",", header=None)

test_df.head()

vals=test_df.values

vals

train_df=pd.read_csv("Desktop/train.csv",delimiter=",", header=None)

train_df.head()

vals=train_df.values

vals

import numpy as np
df_train=train_df.drop([0],axis=0)
df_train.head()

df_test=test_df.drop([0],axis=0)
df_test.head()

train_data = np.array(df_train, dtype = 'float32')
test_data = np.array(df_test, dtype='float32')

x_train = train_data[:,1:]/255

y_train = train_data[:,0]

x_test= test_data[:,1:]/255

y_test=test_data[:,0]

from sklearn.model_selection import train_test_split
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)
%matplotlib inline
import matplotlib.pyplot as plt
%matplotlib notebook
image = x_train[254,:].reshape((28,28))
plt.imshow(image)
plt.show()

image_rows = 28

image_cols = 28

batch_size = 512

image_shape = (image_rows,image_cols,1)



x_train = x_train.reshape(x_train.shape[0],*image_shape)
x_test = x_test.reshape(x_test.shape[0],*image_shape)
x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
cnn_model = Sequential([
    Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),
    MaxPooling2D(pool_size=2) ,
    Dropout(0.5),
    Flatten(), # flatten out the layers
    Dense(32,activation='relu'),
    Dense(10,activation = 'softmax')
    
])

cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])

history = cnn_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=50,
    verbose=1,
    validation_data=(x_validate,y_validate),
)