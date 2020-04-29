# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 00:49:32 2020

@author: Arnob
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import jaccard_similarity_score, classification_report

dataset = pd.read_csv('heart.csv')
# creating input features and target variables
X= dataset.iloc[:,0:13]
y= dataset.iloc[:,13]

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

log_reg = LogisticRegression(C=0.01, solver='liblinear')
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)
pred_proba =  log_reg.predict_proba(X_test)

y_train_pred = log_reg.predict(X_train)
y_test_val=[]
for row,value in y_test.items():
    y_test_val.append(value)
print("Test set: ", log_reg.score(X_test, y_test)*100)
print("Train set: ", log_reg.score(X_train, y_train)*100)
for i in range(20):
	print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], y_test_val[i]))

print('Chances of Heart Disease for patient', pred_proba[:,1])

li = pred_proba[:,1].tolist()

for i in range(len(li)):
    print('Chances of Heart Disease for patient', i+1 , li[i]*100, '%' )

acc_test_set=accuracy_score(y_test, predictions)
print("Test set accuracy: ", acc_test_set*100)

acc_train_set=accuracy_score(y_train, y_train_pred)
print("Train Set accuracy: ",acc_train_set*100)

jacc_test_set=jaccard_similarity_score(y_test, predictions)
print("jaccard similarity score: ", jacc_test_set*100)

log_loss_test_set=log_loss(y_test, pred_proba)
print("log loss: ", log_loss_test_set)

print (classification_report(y_test, predictions))




