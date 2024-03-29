# A Novel Medoid Based Nearest Neghbor Decision Rule (MedNN)
This method is developed in spirit of the nearest neighbor decision rule following a novel medoid based weighting scheme for data classification. The method puts more weightage on the training data that are not only lie close to the test data point, but also lie close to the medoid of its corresponding category in decision making, unlike the standard nearest neighbor algorithms that stress on the data points that are just close to the test data point. The aim of this classifier is to enrich the quality of decision making. The method is explained in the following papers and the steps to implement the method are stated below. The method has performed well for text classification.

[Avideep Mukherjee and Tanmay Basu. A Medoid Based Weighting Scheme for Nearest Neighbors Decision Rule Towards Effective Text Categorization. Springer Nature Applied Sciences, 2020](https://link.springer.com/content/pdf/10.1007/s42452-020-2738-8.pdf).


[Avideep Mukherjee and Tanmay Basu. An Effective Nearest Neighbor Classification Technique Using Medoid Based Weighting Scheme, published in Proceedings of the Fourteenth International Conference on Data Science, pp.231-234, Las Vegas, USA, 2018](https://csce.ucmss.com/cr/books/2018/LFS/CSREA2018/ICD8039.pdf).

## Prerequisites
[Python 3](https://www.python.org/downloads/), [Copy](https://docs.python.org/3/library/copy.html), [NumPy](https://numpy.org/install/), [Scipy](https://pypi.org/project/scipy/), [Scikit-Learn](https://scikit-learn.org/0.16/install.html)

## How to run the model?

The model is implemented in `MedNN.py`. Run the following lines to train the classifier on a set of data samples and subsequently test it's performance on another set of data samples. 

```
clf = MedNN(gamma=0.025,L=3,metric='cosine')
clf.fit(X_train,y_train)
predicted_class_label = clf.predict(X_test)
```

Here `X_train` is the training data and it is a numeric array or matrix and has shapes '[n_samples, n_features]'. `y_train` is the class labels of individual samples in X_train. Similarly, `X_test` is the test data and it is also an array or matrix and has shapes '[n_samples, n_features]'. The following options of `distance metrics` are available: 'cosine', 'chebyshev', 'cityblock', 'euclidean', 'minkowski' and the `default` is `cosine distance`. `L` is the threshold on majority voting and `gamma` is the threshold on difference of weights between two classes for a test data point. The value of gamma is recommended to be close to zero.

An example code to execute `MedNN.py` is uploaded as `Testing_MedNN.py`. 

## Contact

For any further query, comment or suggestion, you may reach out to me at welcometanmay@gmail.com or Avideep Mukherjee at mukherjeeavideep@gmail.com

## Citation
```
@article{mukherjee20mednntext,
	title={A medoid-based weighting scheme for nearest-neighbor decision rule toward effective text categorization},
	author={A. Mukherjee and T. Basu},
	journal={SN Applied Sciences},
	volume={2},
	pages={1--9},
	year={2020},
	publisher={Springer}
}

@inproceedings{mukherjee18mednn,
 author    = "A. Mukherjee and T. Basu ",
 title     = "An Effective Nearest Neighbor Classification Technique Using Medoid Based Weighting Scheme ",
 year      = "2018 ",
 pages     = "231-234 ",
 editor    = " ",
 booktitle = "Proceedings of International Conference on Data Science ",
 address   = " ",
 publisher = "CSREA Press "
}
```
