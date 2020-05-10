from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from MedNN import MedNN 


# Lodaing a dataset

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifier calling

clf = MedNN(gamma=0.025,L=2)
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
