#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys

sys.path.append("../tools/")
from tools.email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
from sklearn import tree

clf_40 = tree.DecisionTreeClassifier(min_samples_split=40)
clf_40 = clf_40.fit(features_train, labels_train)

accuracy_clf_40 = clf_40.score(features_test, labels_test)
print(accuracy_clf_40)


#########################################################


