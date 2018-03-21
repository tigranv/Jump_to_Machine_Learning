from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

string1 = "hi Katie the self driving car will be late best Sebastian"
string2 = "hi Sebastian the machine learning class will be great great great best Katie"
string3 = "hi Katie the machine learning class will be most excellent"

email_list = [string1, string2, string3]

bag_of_words = vectorizer.fit(email_list)
bag_of_words = vectorizer.transform(email_list)
print(bag_of_words)
print("------------------------------------------------")

print(vectorizer.vocabulary_.get("great"))

from nltk.corpus import stopwords
sw = stopwords.words("english")
print(len(sw))

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
print(stemmer.stem("responsiveness"))
print(stemmer.stem("unresponsive"))


