#!/usr/bin/python

import random
import numpy
import matplotlib.pyplot as plt
import pickle


### load up some practice data with outliers in it
ages = pickle.load( open("practice_outliers_ages.pkl", "r") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "r") )
print("1")


### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like

# net worth is the target and the feature being used to predict it is a person's
# age (remember to train on the training data!).

from sklearn import linear_model

reg = linear_model.LinearRegression()
reg = reg.fit(ages_train, net_worths_train)

slope = reg.coef_
intercept = reg.intercept_
test_score = reg.score(ages_test, net_worths_test)
training_score = reg.score(ages_train, net_worths_train)

print(slope,intercept,test_score, training_score)
# (array([[ 5.07793064]]), array([ 25.21002155]), 0.87826247036646732, 0.48987259617514989)

try:
    plt.plot(ages, reg.predict(ages), color="blue")
    print("plot")
except NameError:
    print("error")
    pass
plt.scatter(ages, net_worths)

from outlier_cleaner import outlierCleaner

print ("hello")

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    print("this")
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
    print(len(cleaned_data))
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"

### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        print "slope (w/o outliers):", reg.coef_
        print "r^2 (test w/o outliers)", reg.score(ages_test, net_worths_test)
        plt.plot(ages, reg.predict(ages), color="blue")
        # slope (w/o outliers): [[ 6.36859481]]
        # r^2 (test w/o outliers) 0.983189455396
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")

    print("cleaned_data")
else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"

plt.show() #plot blocks execution!!
