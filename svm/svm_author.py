#!/usr/bin/python

"""

   Hopefully it’s becoming clearer what Sebastian meant when he said Naive Bayes
   is great for text--it’s faster and generally gives better performance than an
   SVM for this particular problem. Of course, there are plenty of other problems
   where an SVM might work better. Knowing which one to try when you’re tackling a
   problem for the first time is part of the art and science of machine learning.
   In addition to picking your algorithm, depending on which one you try,
   there are parameter tunes to worry about as well, and the possibility of overfitting
   (especially if you don’t have lots of training data).

   Our general suggestion is to try a few different algorithms for each problem.
   Tuning the parameters can be a lot of work, but just sit tight for now--toward
   the end of the class we will introduce you to GridCV, a great sklearn tool that
   can find an optimal parameter tune almost automatically.

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