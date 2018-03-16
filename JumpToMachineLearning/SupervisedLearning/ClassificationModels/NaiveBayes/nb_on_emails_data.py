
""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import datetime
sys.path.append("../JumpToMachineLearning/SupervisedLearning/HelperModules/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print("-----------------------------------------------------")

#########################################################
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
classify = GaussianNB()

t0 = time()
print("Fitting starts  at {}".format(datetime.datetime.now().time()))
classify.fit(features_train, labels_train)
print("Fitting finished at {},  training time: {}s".format(datetime.datetime.now().time(), round(time()-t0, 3)))

print("-----------------------------------------------------")

t0 = time()
print("Predicting starts  at {}".format(datetime.datetime.now().time()))
pred = classify.predict(features_test)
print("Predicting finished at {},  training time: {}".format(datetime.datetime.now().time(), round(time()-t0, 3)))

print("-----------------------------------------------------")

accuracy = accuracy_score(labels_test, pred)

print("Accuracy is : {}".format(accuracy))
#########################################################


