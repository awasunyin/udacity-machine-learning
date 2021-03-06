import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## DECISION TREE #################################
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2)
clf = clf.fit(features_train, labels_train)

clf2 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=50)
clf2 = clf2.fit(features_train, labels_train)

acc_min_samples_split_2 = clf.score(features_test, labels_test, sample_weight=None)
acc_min_samples_split_50 = clf2.score(features_test, labels_test, sample_weight=None)


### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively


def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}

"""Good job! Your output matches our solution.
Here's your output:
{'acc_min_samples_split_50': 0.912, 'acc_min_samples_split_2': 0.908}

The simpler decision boundary gives a better accuracy"""