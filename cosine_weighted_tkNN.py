#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:04:01 2018

@author: avideep
"""
import numpy as np
import csv
from scipy.spatial import distance
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy

fread = open('/run/media/avideep/Films/Important/MSc_Project/spambase_data.csv','rb')
data = list(csv.reader(fread,delimiter = ','))
fread.close()
y = [item[57] for item in data]

X = data
for row in X:
    del row[57]   

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
class_names = list(np.unique(y_train))
class_names_dict = {}
for i,val in enumerate(class_names):
    class_names_dict[i] = val

def findMedians(X_train,y_train):
    medians = {}
    for class_name in class_names:
        class_name_x_train = [X_train[i] for i,val in enumerate(y_train) if val == class_name]
        mean = []
        for i in range(len(class_name_x_train[1])):
            col = []
            for row in class_name_x_train:
                col.append(row[i])
            mean.append(np.mean(np.array(col).astype(np.float)))
        median_row = deepcopy(class_name_x_train[0])
        for row in class_name_x_train:
            if(distance.cosine(np.array(row).astype(np.float),np.array(mean).astype(np.float))>distance.cosine(np.array(median_row).astype(np.float),np.array(mean).astype(np.float))):
                median_row = deepcopy(row)
        print 'Median computed for class'
        medians[class_name] = np.array(median_row).astype(np.float)
    return medians

def findSelectedSamples(X_train,y_train,x_test,medians):
    X_train = np.array(X_train).astype(np.float)
    x_test = np.array(x_test).astype(np.float)
    x_train_selected = []
    y_train_selected = []
    for i in range(len(X_train)):
        if(distance.cosine(X_train[i],x_test)<=distance.cosine(medians[y_train[i]],x_test)):
            x_train_selected.append(X_train[i])
            y_train_selected.append(y_train[i])
    return np.array(x_train_selected), y_train_selected

def sortneighbors(x,y,x_test):
    x = np.array(x).astype(np.float)
    x_test = np.array(x_test).astype(np.float)
    dist = np.empty(len(x))
    for i in range(len(x)):
        dist[i] = distance.cosine(x[i],x_test)
    dist = np.argsort(dist)
    x_sorted = np.empty(shape = (len(x),len(X_train[1])))
    y_sorted = []
    k = 0
    for i in dist:
        x_sorted[k] = x[i]
        y_sorted.append(y[i])
        k = k + 1
    return x_sorted,y_sorted

def cosine_weighted_tkNN(X_train,y_train,x_test,medians):
    y_test = '-1'
    x_train_selected, y_train_selected = findSelectedSamples(X_train,y_train,x_test,medians) 
    x_train_sorted, y_train_sorted = sortneighbors(x_train_selected,y_train_selected,x_test)
    x_test = np.array(x_test).astype(np.float)  
    x_train_sorted = np.array(x_train_sorted).astype(np.float)
    L = 2
    while(L <= len(x_train_sorted)):
        S_L = x_train_sorted[:L]
        S_L_y = y_train_sorted[:L]
        weights = np.empty(len(S_L))
        for i in range(len(S_L)):
            weights[i] = distance.cosine(S_L[i],x_test)*distance.cosine(medians[S_L_y[i]],x_test)
        class_weights = np.zeros(len(class_names))
        p = 0
        for class_name in class_names:
            class_weights[p] = np.sum([weights[k] for k in np.where(S_L_y == class_name)])
            p = p + 1 
        max1 = max(class_weights)
        class_weights_temp = list(class_weights)
        class_weights_temp.remove(max1)
        max2 = max(class_weights_temp)
        if(max1 - max2 >=0):
            y_test = class_names[class_weights_temp.index(max1)]
            break
        else:
            L = L + 1
    return y_test
medians = findMedians(X_train,y_train)
predicted_class_label = list(np.empty(len(X_test)))
for i in range(len(X_test)):
    predicted_class_label[i] = cosine_weighted_tkNN(X_train,y_train,X_test[i],medians)
print 'Classification of the test samples'

predicted_class_label = list(predicted_class_label)

print 'Evaluation using Precision, Recall and F-measure'  
pr=precision_score(y_test, predicted_class_label, average='micro')
print '\n Precision:'+str(pr)
re=recall_score(y_test, predicted_class_label, average='micro')
print '\n Recall:'+str(re)
fm=f1_score(y_test, predicted_class_label, average='micro') 
print '\n F-measure:'+str(fm)

#Weighted kNN Classifier :~
neigh = KNeighborsClassifier(weights='distance')
neigh.fit(X_train,y_train)
predicted_class_label_kNN = neigh.predict(X_test)
print 'Classification of the test samples'

predicted_class_label_kNN = list(predicted_class_label_kNN)

print 'Evaluation using Precision, Recall and F-measure'  
pr=precision_score(y_test, predicted_class_label_kNN, average='micro')
print '\n Precision:'+str(pr)
re=recall_score(y_test, predicted_class_label_kNN, average='micro')
print '\n Recall:'+str(re)
fm=f1_score(y_test, predicted_class_label_kNN, average='micro') 
print '\n F-measure:'+str(fm)

