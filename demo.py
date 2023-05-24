from model import ANN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

test=pd.read_csv("mnist_test.csv")
train=pd.read_csv("mnist_train.csv")
X_train=train.iloc[:,1:]
Y_train=train.iloc[:,:1]
X_test=test.iloc[:,1:]
Y_test=test.iloc[:,:1]
X_train=X_train.to_numpy()
Y_train=Y_train.to_numpy()
X_test=X_test.to_numpy()
Y_test=Y_test.to_numpy()

all_layers=input("Enter nodes in each layer separated using commas")
l=[int(i) for i in all_layers.split(',')]

dnn=ANN(4,l,lr=0.01)
dnn.train(X_train,Y_train)

predictions,accuracy=dnn.predict(X_test,Y_test)
print(f"The accuracy of the model on the test data is: {accuracy*100}% ")
