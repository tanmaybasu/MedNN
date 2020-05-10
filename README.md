# medoid-weighted-kNN
Implementation of the algorithm in the paper entitled "[A medoid‑based weighting scheme for nearest‑neighbor decision rule
toward efective text categorization](https://link.springer.com/content/pdf/10.1007/s42452-020-2738-8.pdf)"

# How to run the model?

The model is implemented in MedNN.py. Run the following lines to train the classifier on a set of data samples and subsequently test it's performance on another set of data samples. 

`clf = MedkNN(gamma=0.025,L=3)
clf.fit(X_train,y_train)
predicted_class_label = clf.predict(X_test)`

Here `X_train` is the training data and it is an array or matrix and has shapes `[n_samples, n_features]`. `y_train` is the class labels of individual samples in `X_train`. Similarly, `X_test` is the test data and it is also an array or matrix and has shapes `[n_samples, n_features]`. 

An example code to implement MedNN for text data is uploaded as `testing.py`. For any further query, you may reach out to me at mukherjeeavideep@gmail.com
