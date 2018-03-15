import sys
sys.path.append("../KatieExamples/HelpersAndData/")

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, Show_Image
from ClassifyHelper import Accuracy

picture_name = "NBclf.png"

features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## Naive Bayes #################################

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)

#### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test, picture_name)

Show_Image(picture_name)

print(Accuracy(clf, features_test, labels_test))




    



