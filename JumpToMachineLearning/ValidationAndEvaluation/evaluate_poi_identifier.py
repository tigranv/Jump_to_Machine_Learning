#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../JumpToMachineLearning/Helpers/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../JumpToMachineLearning/AppData/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../JumpToMachineLearning/AppData/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
accuracy =  clf.score(features_test,labels_test)
print(accuracy)
print(sum(labels_test), len(labels_test))

print("---------------------------------------")

pred = clf.predict(features_test)

for i in range(len(pred)):
    print("Predicted {},  Actual {}".format(pred[i], labels_test[i]))

from sklearn.metrics import recall_score, precision_score
print("Recall score is {}".format(recall_score(labels_test, pred)))
print("Precision score is {}".format(precision_score(labels_test, pred)))



