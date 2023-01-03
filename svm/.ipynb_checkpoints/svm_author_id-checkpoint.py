#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

from sklearn.svm import SVC
# clf = SVC(kernel="linear")
clf = SVC(C=10000.0, kernel="rbf")

### To speed up training and predicting time,
### eventually slice down the training dataset
### to alpha percent of its original size
alpha = 0.01
features_train = features_train[:int(len(features_train)*alpha)]
labels_train = labels_train[:int(len(labels_train)*alpha)]

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

### fit the classifier on the training features and labels
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

### calculate and return the accuracy on the test data
t0 = time()
accuracy = clf.score(features_test, labels_test)
print("Predicting Time:", round(time()-t0, 3), "s")

### print out the result
print(f'The accuracy of the Linear Support Vector Machine classifier was calculated as {round(100*accuracy, 4)}%.')

### print some classfications
pred = clf.predict(features_test)
dict_cls = {0: 'Sara', 1: 'Chris'}
for idx in (10, 26 ,50):
    print(f'Element {idx} of the test set corresponds to {dict_cls[pred[idx]]} (== class {pred[idx]}).')

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
