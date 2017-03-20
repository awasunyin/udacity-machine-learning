#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import pandas as pd
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

"""Identifying and cleaning away outliers is something you should always think about
when looking at a data set for the first time, and now you'll get some hands-on experience
with the Enron data.

Since there are two features being extracted from the dictionary ("salary" and "bonus"), the resulting
numpy array will be of dimension N x 2, where N is the number of data points and 2 is the number
of features. This is perfect input for a scatterplot; we'll use the matplotlib.pyplot module to
make that plot.
"""

data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

data1=pd.DataFrame(data_dict)
print data1.ix['salary'].idxmax(axis=1)
# TOTAL

max_salary = 0
max_salary_key = None

for key in data_dict:
	if data_dict[key]["salary"] != 'NaN' and data_dict[key]["salary"] > max_salary:
		max_salary = data_dict[key]["salary"]
		max_salary_key = key

data_dict.pop(max_salary_key, 0)

# Who made $6-8 mil bonus and over $1 mil salary

for key in data_dict:
	if data_dict[key]["salary"] != 'NaN' and data_dict[key]["bonus"] != 'NaN':
		if data_dict[key]["salary"] > 1.0e6 and data_dict[key]["bonus"] > 5.5e6:
			print key

features = ["salary", "bonus"]

data = featureFormat(data_dict, features)
# LAY KENNETH L
# SKILLING JEFFREY K


# visualisation to find outliers

for point in data:  
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



