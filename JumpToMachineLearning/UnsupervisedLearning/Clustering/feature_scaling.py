#from sklearn.preprocessing import MinMaxScaler
#import numpy
#weights = numpy.array([[115.], [140.], [175.]])
#scaler = MinMaxScaler()
#scaled_weights = scaler.fit_transform(weights)

#print(scaled_weights)

import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../JumpToMachineLearning/Helpers/")
from feature_format import featureFormat, targetFeatureSplit
from class_vis import Draw

### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load(open("../JumpToMachineLearning/AppData/final_project_dataset.pkl",  "rb"))
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
finance_features = scl.fit_transform(finance_features)
import numpy as np
features_test = np.array([[200000., 1000000.]])
print(scl.transform(features_test) )


from sklearn.cluster import KMeans
clf = KMeans(n_clusters=2)
pred = clf.fit_predict(finance_features)


try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters_scaled.png", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("no predictions object named pred found, no clusters to plot")

