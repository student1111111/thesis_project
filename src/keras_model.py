from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
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
m.add(Dense(units=512,activation = 'relu'))
m.add(Dropout(0.5))
m.add(Dense(units=512,activation = 'relu', input_dim = 21))
m.add(Dropout(0.5))
m.add(Dense(units=512,activation = 'relu', input_dim = 21))
m.add(Dropout(0.5))
m.add(Dense(units=10,activation = 'softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
              model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
              model.fit(x_train, y_train, epochs=5, batch_size=32)
              model.train_on_batch(x_batch, y_batch)
              loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
              classes = model.predict(x_test, batch_size=128)
