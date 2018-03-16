import sys
sys.path.append("../JumpToMachineLearning/SupervisedLearning/HelperModules/")

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, Show_Image
from ClassifyHelper import Accuracy

features_train, labels_train, features_test, labels_test = makeTerrainData()

from sklearn.neighbors import KNeighborsClassifier
acurr_list = []
n_neighbors_list = range(1, 250, 1)

for x in n_neighbors_list:
    clf = KNeighborsClassifier(n_neighbors=x)
    clf.fit(features_train, labels_train)
    acurr_list.append(Accuracy(clf, features_test, labels_test))



import matplotlib.pyplot as plt
plt.plot(n_neighbors_list, acurr_list)
#plt.axis([0, 6, 0, 20])
plt.show()