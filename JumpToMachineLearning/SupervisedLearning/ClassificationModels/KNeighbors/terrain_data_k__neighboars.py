import sys
sys.path.append("../JumpToMachineLearning/Helpers/")

from prep_data import makeTerrainData
from class_vis import prettyPicture, Show_Image
from ClassifyHelper import Accuracy

picture_name = "K_neighbclf.png"

features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## k nearest neighbors #################################

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=8)
clf.fit(features_train, labels_train)

#### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test, picture_name)

Show_Image(picture_name)

accuracy = Accuracy(clf, features_test, labels_test)

print("Accuracy score of NB for terrain data is : {}".format(accuracy))



### initial visualization
#grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
#bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
#grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
#bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


##### initial visualization
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
#plt.show()

