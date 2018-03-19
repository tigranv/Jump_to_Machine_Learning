
""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import datetime
sys.path.append("../JumpToMachineLearning/Helpers/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print("-----------------------------------------------------")

#features_train = features_train[:(int)(len(features_train)/100)] 
#labels_train = labels_train[:(int)(len(labels_train)/100)] 
#########################################################
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#classify = SVC(C = 10000, kernel="linear")
classify = SVC(C = 10000, kernel="rbf")

t0 = time()
print("Fitting starts  at {}".format(datetime.datetime.now().time()))
classify.fit(features_train, labels_train)
print("Fitting finished at {},  training time: {}s".format(datetime.datetime.now().time(), round(time()-t0, 3)))

print("-----------------------------------------------------")

t0 = time()
print("Predicting starts  at {}".format(datetime.datetime.now().time()))
pred = classify.predict(features_test)
print("Predicting finished at {},  training time: {}".format(datetime.datetime.now().time(), round(time()-t0, 3)))

accuracy = accuracy_score(labels_test, pred)

print("-----------------------------------------------------")

print("Accuracy is : {}".format(accuracy))

print("For 10 is {}, For 26 is {}, For 50 is {},".format(pred[10], pred[26], pred[50]))

print(sum(pred))

#########################################################



