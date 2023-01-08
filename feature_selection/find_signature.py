#!/usr/bin/python3

import joblib
import numpy
numpy.random.seed(42)

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = joblib.load(words_file)
authors = joblib.load(authors_file)



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

print(f"The number of training points is limited to: {len(features_train):,}")

### your code goes here

### import the sklearn module for Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

### create classifier
clf = DecisionTreeClassifier(random_state=42)

### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)

### use the trained model to predict new values
pred = clf.predict(features_train)

### calculate the accuracy
accuracy = accuracy_score(pred, labels_train)
print(f'The training accuracy of the Decision Tree classifier was calculated as {round(100*accuracy, 2)}%.')

### use the trained model to predict new values
pred = clf.predict(features_test)

### calculate the accuracy
accuracy = accuracy_score(pred, labels_test)
print(f'The test accuracy of the Decision Tree classifier was calculated as {round(100*accuracy, 2)}%.')

### get the most important features
vocab = vectorizer.get_feature_names_out()
if max(clf.feature_importances_) <= .2:
    print("No feature has an importance score higher than 0.2.")
else:
    for idx, score in enumerate(clf.feature_importances_):
        if score > .2:
            print(f"Feature #{idx} has an importance score of {round(score, 4)}")
            print(f"Feature '{vocab[idx]}' has an importance score of {round(score, 4)}")


