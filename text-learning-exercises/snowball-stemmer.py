from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
r = stemmer.stem("responsiveness")
print(r)