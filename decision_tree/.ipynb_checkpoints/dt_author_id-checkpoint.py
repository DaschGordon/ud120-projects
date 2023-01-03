#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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

print(f'The number of features in the training data is: {len(features_train[0])}')


#########################################################
### your code goes here ###
   
### import the sklearn module for Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

### create classifier
min_samples_split = 40
clf = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=0)

### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)

### use the trained model to predict new values
pred = clf.predict(features_test)

### calculate the accuracy
accuracy = accuracy_score(pred, labels_test)
print(f'The accuracy of the Decision Tree classifier was calculated as {round(100*accuracy, 4)}%.')

#########################################################


