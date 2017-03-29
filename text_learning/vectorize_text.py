#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

# Remember to run setup.py for this project!


from_sara = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
#temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        #temp_counter += 1
        #if temp_counter < 200:
            path = os.path.join('..', path[:-1])
            print path
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
            text = parseOutText(email)

            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            for signature_word in ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]:
                text = text.replace(signature_word, '')

            ### append the text to word_data
            word_data.append(text)

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == "sara":
                from_data.append(0)
            else:
                from_data.append(1)

            email.close()

print "emails processed"
print word_data[152]
# word_data[152] --> tjonesnsf stephani and sam need nymex calendar
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )


### in Part 4, do TfIdf vectorization here
"""In vectorize_text.py, you will iterate through all the emails from
Chris and from Sara. For each email, feed the opened email to parseOutText()
and return the stemmed text string. Then do two things:

remove signature words ("sara", "shackleton", "chris", "germani"--bonus points
if you can figure out why it's "germani" and not "germany")
append the updated text string to word_data -- if the email is from Sara, append 0
(zero) to from_data, or append a 1 if Chris wrote the email.
Once this step is complete, you should have two lists: one contains the stemmed text
of each email, and the second should contain the labels that encode (via a 0 or 1)
who the author of that email is.

Running over all the emails can take a little while (5 minutes or more), so we've added
a temp_counter to cut things off after the first 200 emails. Of course, once everything
is working, you'd want to run over the full dataset.

In the box below, put the string that you get for word_data[152].
START QUIZ
"""

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = 'english')
vectorizer.fit(word_data)
print len(vectorizer.get_feature_names())
print vectorizer.get_feature_names()[34597]
#with first 200 = 3078
# 38757
# stephaniethank

# tjonesnsf stephani and sam need nymex calendar
# 38756
# stephanlonect
