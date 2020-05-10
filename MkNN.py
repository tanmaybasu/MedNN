import os,sys
import numpy as np
import scipy.sparse as sp
from scipy.spatial import distance
from collections import Counter, defaultdict
from scipy.sparse import csc_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split, GridSearchCV
from copy import deepcopy
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif 



class MkNN():
    def __init__(self,gamma = 0.025,L=2):
        self.gamma = gamma
        self.L = L
    def findMedians(self,X_train,y_train,class_names):
        medians = {}
        for class_name in class_names:
            class_name_x_train = [X_train[i] for i,val in enumerate(y_train) if val == class_name]
            mean = []
            for i in range(len(class_name_x_train[0])):
                col = []
                for row in class_name_x_train:
                    col.append(row[i])
                mean.append(np.mean(np.array(col).astype(np.float)))
            median_row = deepcopy(class_name_x_train[0])
            for row in class_name_x_train:
                if distance.cosine(np.array(row).astype(np.float),np.array(mean).astype(np.float)) < distance.cosine(np.array(median_row).astype(np.float),np.array(mean).astype(np.float)):
                    median_row = deepcopy(row)
            medians[class_name] = np.array(median_row).astype(np.float)
        return medians
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.class_names = list(np.unique(self.y_train))
        self.class_names_dict = {}
        for i,val in enumerate(self.class_names):
            self.class_names_dict[i] = val
        self.medians = self.findMedians(self.X_train,self.y_train,self.class_names)
    def findSelectedSamples(self,X_train,y_train,x_test,medians,class_names,class_names_dict):
        X_train = np.array(self.X_train).astype(np.float)
        x_test = np.array(x_test).astype(np.float)
        x_train_selected = []
        y_train_selected = []
        median_list = list(np.empty(len(class_names)))
        for i in range(len(class_names)):
            median_list[i] = distance.cosine(medians[class_names_dict[i]],x_test)
        min_median = max(median_list)
        for i in range(len(X_train)):
            if(distance.cosine(X_train[i],x_test)<=distance.cosine(min_median,x_test)):
            #if(distance.cosine(X_train[i],x_test)<=distance.cosine(medians[y_train[i]],x_test)):
                x_train_selected.append(X_train[i])
                y_train_selected.append(y_train[i])
        return np.array(x_train_selected), y_train_selected
 
    def sortneighbors(self,x,y,X_train,x_test):
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
    def predict(self,X_test):
        Y_test = np.empty(X_test.shape[0])
        for ii in range(X_test.shape[0]):
            Y_test[ii] = -1
            x_test = X_test[ii] 
            x_train_selected, y_train_selected = self.findSelectedSamples(self.X_train,self.y_train,x_test,
                                                                    self.medians,self.class_names,self.class_names_dict)
            
            x_train_sorted, y_train_sorted = self.sortneighbors(x_train_selected,y_train_selected,self.X_train,x_test)
            x_test = np.array(x_test).astype(np.float)  
            x_train_sorted = np.array(x_train_sorted).astype(np.float)
            L = self.L
        #    scaler = MinMaxScaler()
            no_of_neighbors=len(x_train_sorted) 
            while(L <= no_of_neighbors):
                S_L = x_train_sorted[:L]        #Sorted neighbors
                S_L_y = y_train_sorted[:L]      #Class labels of the sorted neighbors
                weights = np.empty(len(S_L))
                weights_temp = []
                for i in range(len(S_L)):
                    weights_temp.append([distance.cosine(S_L[i],x_test), distance.cosine(self.medians[S_L_y[i]],S_L[i])])
                for i in range(len(S_L)):
                    weights[i] = ((1 - distance.cosine(S_L[i],x_test))*(1 - distance.cosine(self.medians[S_L_y[i]],x_test))) #Old version
                class_weights = np.zeros(len(self.class_names))
                class_cardinality = np.zeros(len(self.class_names))
                p = 0
                for class_name in self.class_names:
                    class_weights[p] = np.sum([weights[k] for k in np.where(S_L_y == class_name)])
                    class_cardinality[p]=0
                    for k in S_L_y:
                        if k==class_name:
                            class_cardinality[p]=class_cardinality[p]+1
                    p = p + 1
                max1 = max(class_weights)
                max1_index = np.argmax(class_weights)
                if class_cardinality[max1_index] == 0:
                    nmax1 = 0
                else:
                    nmax1 = max1 /class_cardinality[max1_index]
                class_weights_max_containing = list(class_weights)
                class_weights_temp = list(class_weights)
                class_weights_temp.remove(max1)
                max2 = max(class_weights_temp)
                max2_index = np.argmax(class_weights_temp)
                if class_cardinality[max2_index] == 0:
                    nmax2 = 0
                else:
                    nmax2 = max2 /class_cardinality[max2_index]
                if(nmax1 - nmax2 >=self.gamma):
                    Y_test[ii] = self.class_names[class_weights_max_containing.index(max1)]
                    break
                    #return y_test, L
                else:
                    L = L + 1
        return Y_test