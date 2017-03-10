#!/usr/bin/python

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

from sklearn import svm
from sklearn.metrics import accuracy_score

#Changed 'linear' to 'rbf'
linear_kernel_svm = svm.SVC(kernel='linear', C=500.0)

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

t0 = time()
linear_kernel_svm.fit(features_train, labels_train)
print "training time with SVM's linear kernel", time() - t0

t1 = time()
pred = linear_kernel_svm.predict(features_test)
print "prediction time with SVM's linear kernel", time() - t1

accuracy = accuracy_score(labels_test, pred)
print accuracy

#########################################################