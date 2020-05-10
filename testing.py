from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from helper_py import load_karypis
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from MkNN import MkNN 


# # Karypis



X,y,number_of_class_labels = load_karypis('tr45',path="karypis",min_df=3)
sss = StratifiedShuffleSplit(n_splits=1,test_size=0.20,random_state=1)
train_index, test_index = next(sss.split(X,y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
tfidf = TfidfTransformer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
X_train, X_test = X_train.toarray(), X_test.toarray()

# Classifier calling

clf = MkNN(gamma=0.025,L=2)
clf.fit(X_train,y_train)
predicted_class_label = clf.predict(X_test)

# Evaluation

print ('Classification of the test samples')
print ('Evaluation using Precision, Recall and F-measure (micro)')  
pr=precision_score(y_test, predicted_class_label, average='micro')
print ('\n Precision:'+str(pr))
re=recall_score(y_test, predicted_class_label, average='micro')
print ('\n Recall:'+str(re))
fm=f1_score(y_test, predicted_class_label, average='micro') 
print ('\n Mircro Averaged F1-Score:'+str(fm))

print ('Evaluation using Precision, Recall and F-measure (macro)')
pr=precision_score(y_test, predicted_class_label, average='macro')
print ('\n Precision:'+str(pr))
re=recall_score(y_test, predicted_class_label, average='macro')
print ('\n Recall:'+str(re))
fm=f1_score(y_test, predicted_class_label, average='macro') 
print ('\n Macro Averaged F1-Score :'+str(fm))
acc=accuracy_score(y_test, predicted_class_label) 
print ('\n Accuracy Score:'+str(acc))
