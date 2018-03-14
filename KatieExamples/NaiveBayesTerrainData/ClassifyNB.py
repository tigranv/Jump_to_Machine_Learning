def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    
    ### your code goes here!
    from sklearn.naive_bayes import GaussianNB
    classify = GaussianNB()

    classify.fit(features_train, labels_train)
    return classify

def NBAccuracy(clf, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    
    from sklearn.metrics import accuracy_score

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = accuracy_score(labels_test, pred)
    return accuracy
