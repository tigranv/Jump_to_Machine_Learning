import sys
sys.path.append("../KatieExamples/HelpersAndData/")

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, Show_Image
from ClassifyHelper import Accuracy

picture_name = "SVMclf.png"

features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## SVM #################################

from sklearn.svm import SVC
clf = SVC(C=1000.0, kernel="rbf", gamma = 10) 
#  large C means more trainig points uncluded, 
#  kernel  see in sklearn documentation
#  gamma defines how far the influence of a single trainig example reaches, high gamma means very curvy decision boundary

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

prettyPicture(clf, features_test, labels_test, picture_name)

Show_Image(picture_name)

accuracy = Accuracy(clf, features_test, labels_test)

print("Accuracy score of svm for terrain data is : {}".format(accuracy))
