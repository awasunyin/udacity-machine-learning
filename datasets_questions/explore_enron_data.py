#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# Q1: How many data points (people)? = 146
print len(enron_data)

# Q2: Per person, how many features? = 21
print len(enron_data["SKILLING JEFFREY K"])

# Q3: How many POIs are there in the E+F dataset? = 18
print len(dict((key, value) for key, value in enron_data.items() if value["poi"] == True))

# Q4: How many POIs are there in total? = 35
poi_reader = open('../final_project/poi_names.txt', 'r')
poi_reader.readline() # skip url
poi_reader.readline() # skip blank line

poi_count = 0
for poi in poi_reader:
	poi_count += 1

print poi_count

# Q5: What might be a problem with having some POIs missing from our dataset?
"""There are a few things you could say here, but our main thought
is about having enough data to really learn the patterns.
In general, more data is always better--only having 18 data
points doesn't give you that many examples to learn from."""

# Q6: # What is the total value of the stock belonging to James Prentice? = 1095040
print enron_data["PRENTICE JAMES"]["total_stock_value"]

# Q7: How many email messages do we have from Wesley Colwell to persons of interest? = 11
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

# Q8: Whats the value of stock options exercised by Jeffrey K Skilling? = 19250000
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

# Q10: Among Lay, Skilling and Fastow, who took home the most money?
most_paid = ''
highest_payment = 0

for key in ('LAY KENNETH L', 'FASTOW ANDREW S', 'SKILLING JEFFREY K'):
	if enron_data[key]['total_payments'] > highest_payment:
		highest_payment = enron_data[key]['total_payments']
		most_paid = key

print most_paid, highest_payment

# Q11: How is it denoted when a feature doesn't have a well-defined value?
print(enron_data['SKILLING JEFFREY K'])

# Q12: How many folks have a quantified salary?
print len(dict((key, value) for key, value in enron_data.items() if value["salary"] != 'NaN'))

# Q12b: How many with a known email address?
print len(dict((key, value) for key, value in enron_data.items() if value["email_address"] != 'NaN'))

# A python dictionary can't be read directly into an sklearn classification
# or regression algorithm; instead, it needs a numpy array or a list of lists
# (each element of the list (itself a list) is a data point, and the elements of
# the smaller list are the features of that point).
# We've written some helper functions (featureFormat() and targetFeatureSplit()
# in tools/feature_format.py) that can take a list of feature names and the data
# dictionary, and return a numpy array.
# In the case when a feature does not have a value for a particular person, this
# function will also replace the feature value with 0 (zero)."""

#  Q13: How many people have NaN for total_payments? What is the percentage of total? = 14.38%
no_total_payments = len(dict((key, value) for key, value in enron_data.items() if value["total_payments"] == 'NaN'))
print float(no_total_payments)/len(enron_data) * 100

#  Q14: What percentage of POIs in the data have "NaN" for their total payments? 0%
POIs = dict((key,value) for key, value in enron_data.items() if value['poi'] == True)
number_POIs = len(POIs)
no_total_payments = len(dict((key, value) for key, value in POIs.items() if value["total_payments"] == 'NaN'))
print float(no_total_payments)/number_POIs * 100

#  Q15: If 10 POIs with NaN total_payments were added, what is the new number of people? = 156
# What is the new number of people with NaN total_payments? = 31
print len(enron_data) + 10
print 10 + len(dict((key, value) for key, value in enron_data.items() if value["total_payments"] == 'NaN'))

#  Q16: What is the new number of POIs? = 28
print 10 + len(POIs)

#  Q17: What percentage have NaN for their total_payments? = 35.7142857143
print float(10)/(10 + len(POIs))*100
print float(10)/(10 + len(POIs))

count_poi = 0
count_nan = 0

for k in enron_data:
    if enron_data[k]['poi'] == True:  # '== True' can be suppressed
        count_poi += 1
        if enron_data[k]['total_payments'] == "NaN":
            count_nan += 1

# print count
print float(count_nan) / len(enron_data)

"""This goes to say that, when generating or augmenting a dataset, you should
be exceptionally careful if your data are coming from different sources for different
classes. It can easily lead to the type of bias or mistake that we showed here. There
are ways to deal with this, for example, you wouldn't have to worry about this problem
if you used only email data--in that case, discrepancies in the financial data wouldn't
matter because financial features aren't being used. There are also more sophisticated
ways of estimating how much of an effect these biases can have on your final answer;
those are beyond the scope of this course."""




