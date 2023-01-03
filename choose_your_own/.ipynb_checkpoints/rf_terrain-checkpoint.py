#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

from time import time
from random import shuffle

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_features=None)

### To speed up training and predicting time,
### eventually slice down the training dataset
### to alpha percent of its original size
alpha = 1
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
print(f'The accuracy of the Random Forrest classifier was calculated as {round(100*accuracy, 4)}%.')

### print some classfications
pred = clf.predict(features_test)
dict_cls = {0: 'fast', 1: 'slow'}
list_idx = list(range(50))
shuffle(list_idx)
for idx in list_idx[:5]:
    print(f'Element {idx} of the test set corresponds to {dict_cls[pred[idx]]} (== class {pred[idx]}).')


quit()

try:
    prettyPicture(clf, features_test, labels_test)
    plt.show()
except NameError:
    pass
