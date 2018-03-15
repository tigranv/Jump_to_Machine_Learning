import sys
sys.path.append("../KatieExamples/HelpersAndData/")

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, Show_Image
from ClassifyHelper import Accuracy

picture_name = "DTclf.png"

features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## decision Trees #################################

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 50)
# min_samples_split The minimum number of samples required to split an internal node, how complicated the d.s is

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

prettyPicture(clf, features_test, labels_test, picture_name)

Show_Image(picture_name)

accuracy = Accuracy(clf, features_test, labels_test)

print("Accuracy score of DT for terrain data is : {}".format(accuracy))