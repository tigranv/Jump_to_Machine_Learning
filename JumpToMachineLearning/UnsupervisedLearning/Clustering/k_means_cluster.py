import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../JumpToMachineLearning/Helpers/")
from feature_format import featureFormat, targetFeatureSplit
from class_vis import Draw


### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../JumpToMachineLearning/AppData/final_project_dataset.pkl", "rb"), fix_imports = True )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

salMin = float("inf")
salMax = 0
for k in data_dict:
    sal = data_dict[k]["salary"]
    if sal != 'NaN':
        if sal < salMin:
            salMin = sal
        if sal > salMax:
            salMax = sal

print("min:", salMin)
print("max:", salMax)

### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
#feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2)
plt.show()

from sklearn.cluster import KMeans
clf = KMeans(n_clusters=2)
pred = clf.fit_predict(finance_features)



### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.png", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("no predictions object named pred found, no clusters to plot")
