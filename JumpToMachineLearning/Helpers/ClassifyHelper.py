
def Accuracy(clf, features_test, labels_test):
    
    from sklearn.metrics import accuracy_score

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)

    accuracy = accuracy_score(labels_test, pred)
    return accuracy
