# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 05:13:33 2019

@author: Arnob
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss
dataset = pd.read_csv('heart.csv')
# creating input features and target variables
X= dataset.iloc[:,0:13]
y= dataset.iloc[:,13]

#standardizing the input feature
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(10, activation='relu', kernel_initializer='random_normal', input_dim=13))
#Second  Hidden Layer
classifier.add(Dense(10, activation='relu', kernel_initializer='random_normal'))

#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

_, eval_model=classifier.evaluate(X_train, y_train)


y_test_val=[]
for row,value in y_test.items():
    y_test_val.append(value)

predictions = classifier.predict_classes(X_test)
y_train_pred = classifier.predict_classes(X_train)
pred = classifier.predict_proba(X_test)
li = pred[:,0].tolist()

for i in range(len(li)):
    print('Chances of Heart Disease for patient', i+1 , li[i]*100, '%' )
# summarize the first 5 cases
for i in range(10):
	print(i, '%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], y_test_val[i]))

print("Model Evaluation", eval_model*100)

acc_test_set=accuracy_score(y_test, predictions)
print("Test set accuracy: ", acc_test_set*100)

acc_train_set=accuracy_score(y_train, y_train_pred)
print("Train Set accuracy: ",acc_train_set*100)

log_loss_test_set=log_loss(y_test, pred)
print("log loss: ", log_loss_test_set)