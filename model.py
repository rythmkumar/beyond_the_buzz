import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class ANN:
  def __init__(self,n_layers,*args,lr=0.01):
    self.input_size=args[0][0]
    self.output_size=args[0][-1]
    self.lr=lr
    self.layers=args
    self.n_layers=n_layers
    self.W=[]
    self.A=[]
    self.Z=[]
    self.__random_init()
  def __random_init(self):
    for i in range(self.n_layers-1):
      #  print(self.layers[0][i+1])
      self.W.append(np.random.randn(self.layers[0][i+1],self.layers[0][i])*np.sqrt(1.0/self.layers[0][i+1]))
      self.A.append(np.zeros((self.layers[0][i],1)))
      self.Z.append(np.zeros((self.layers[0][i+1],1)))
    self.A.append(np.zeros((self.layers[0][self.n_layers-1],1)))
  def __sigmoid(self,x,derivative=False):
    if derivative:
      return (np.exp(-x))/((np.exp(-x)+1)**2)
    return 1/(1+np.exp(-x))
  
  def __softmax(self,x,derivative=False):
    exps=np.exp(x-x.max())
    if derivative:
      return exps/np.sum(exps,axis=0)*(1-exps/np.sum(exps,axis=0))
    return exps/np.sum(exps,axis=0)
  def __forward_pass(self,x_train):
    W=self.W
    A=self.A
    Z=self.Z
    A[0]=x_train
    for i in range(self.n_layers-1):
      Z[i]=(np.dot(W[i],A[i]))
      if (i<self.n_layers-2):
        A[i+1]=(self.__sigmoid(Z[i]))
      else:
        A[i+1]=(self.__softmax(Z[i]))
    return A[-1]  

  def __backward_pass(self,y_train,output):
    W=self.W
    A=self.A
    Z=self.Z
    change=[0 for j in range(self.n_layers-1)]
    for i in range(self.n_layers-1):
      if(i==0):      
        error=output-y_train
        change[self.n_layers-i-2]=(np.outer(error,A[self.n_layers-i-2]))
      else:
        error=np.dot(W[self.n_layers-i-1].T,error)*self.__sigmoid(Z[self.n_layers-i-2],derivative=True)
        change[self.n_layers-i-2]=(np.outer(error,A[self.n_layers-i-2]))
    return change

  def __update_weights(self,change):
    for i in range(len(change)):
      self.W[i]-=(self.lr*change[i])


  def predict(self,X_test,Y_test):
    predictions=[]
    j=0
    pred1=[]
    accuracy=0
    for x in X_test:
      y=Y_test[j]
      x=(x/255.0*0.99)+0.01
      targets=np.zeros(10)+0.01
      targets[int(y)]=0.99
      output=self.__forward_pass(x)
      pred=np.argmax(output)
      predictions.append(pred)
      pred1.append(pred==np.argmax(targets))
      j+=1
    accuracy=np.mean(pred1)*100
    return predictions,accuracy
  def train(self,X_train,Y_train,epochs=10):
    start_time=time.time()
    for i in range(epochs):
      j=0
      for x in X_train:
        y=Y_train[j]
        x=(x/255.0*0.99)+0.01
        targets=np.zeros(10)+0.01
        targets[int(y)]=0.99
        output=self.__forward_pass(x)
        change=self.__backward_pass(targets,output)
       
        self.__update_weights(change)
        j+=1

      # _,accuracy=self.predict(X_test,Y_test)
      # print('Epoch: {0}, Time Spent: {1:.02f}s, Accuracy: {2:.02f}%'.format(i+1, time.time()-start_time, accuracy))
