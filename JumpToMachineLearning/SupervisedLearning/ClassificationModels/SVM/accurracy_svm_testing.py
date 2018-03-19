  
import sys
from time import time
import datetime
sys.path.append("../JumpToMachineLearning/Helpers/")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

print("-----------------------------------------------------")

features_train = features_train[:(int)(len(features_train) / 100)] 
labels_train = labels_train[:(int)(len(labels_train) / 100)] 
#########################################################
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


for x in range(1, 5, 1):
    print("------------------------For C = {}-----------------------------".format(10 ** x))
    classify = SVC(C = 10 ** x, kernel="rbf")

    t0 = time()
    classify.fit(features_train, labels_train)
    print("Training time: {}s".format(round(time() - t0, 3)))

    t0 = time()
    pred = classify.predict(features_test)
    print("Predicting time: {}".format(datetime.datetime.now().time(), round(time() - t0, 3)))

    accuracy = accuracy_score(labels_test, pred)

    print("Accuracy is : {}".format(accuracy))

#########################################################



