from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

string1 = "hi katie the self driving car will be late Best Sebastian"
string2 = "hi Sebastian the machine learning class will be great great great Best Katie"
string3 = "hi Katie machine learning class will be most excellent"

email_list = [string1, string2, string3]

bag_of_words = vectorizer.fit(email_list)
bag_of_words = vectorizer.transform(email_list)

print(bag_of_words)
# prints tuples and integers
# In document number 1, word number 7 occurs 1 time --> (1, 7) --> 1
print vectorizer.vocabulary_.get("great")