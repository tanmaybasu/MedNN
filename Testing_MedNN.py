#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday, May 10, 2020 @ 14:37:44

@author: Avideep Mukherjee


"""

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
from sklearn.datasets import load_iris
from MedNN import MedNN


##########################   Lodaing IRIS Data  ##########################  
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


##########################  Classification ##########################  
try:
    gamma=float(input("Enter the value of gamma OR press Enter, if you want to use the default value: "))
    L=int(input("Enter the value of L OR press Enter, if you want to use the default value: "))
    distance=input("Enter the name of the metric: ")  
except:  
    gamma = 0.025; L=2; metric='cosine'
    print('The default values of theta and beta will be used \n') 
clf=MedNN(gamma,L,metric)
clf.fit(X_train,y_train)
predicted_class_label = clf.predict(X_test)

# Evaluation
fm=f1_score(y_test, predicted_class_label, average='macro') 
print ('\n Macro Averaged F1-Score :'+str(fm))
fm=f1_score(y_test, predicted_class_label, average='micro') 
print ('\n Mircro Averaged F1-Score:'+str(fm))



