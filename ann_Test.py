import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding
from matplotlib import pyplot as plt
import tensorflow as tf

ann = Sequential()
ann.add(Dropout(.1, input_shape=(42,)))
ann.add(Dense(20, activation='relu'))
ann.add(Dropout(.1, input_shape=(20,)))
ann.add(Dense(10, activation='relu'))
ann.add(Dense(1, activation="relu"))
# Add output layer
from keras.utils import plot_model

plot_model(ann,
           to_file="model.png",
           show_shapes=True,
           show_layer_names=True,
           )

ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(ann.summary())
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_ann.csv')
df.columns = [i for i in range(df.shape[1])]
print(df)
df = df.rename(columns={42: 'Output'})
print(df)
X = df.iloc[:, :-1]
print("Features shape =", X.shape)

Y = df.iloc[:, -1]
print("Labels shape =", Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
ann.fit(x_train, y_train, batch_size=32 , epochs=20)


from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = ann.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
