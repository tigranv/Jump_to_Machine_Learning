# Loads dataset of enron bonus and salary,  

import sys
import pickle
sys.path.append("../JumpToMachineLearning/Helpers/")
from feature_format import featureFormat, targetFeatureSplit


dictionary = pickle.load( open("../JumpToMachineLearning/AppData/final_project_dataset_modified.pkl", "rb") )

# list the features you want to look at--first item in the list will be the "target" feature
# try to use as a feature long term incentive (features_list = ["bonus", "long_term_incentive"])
features_list = ["bonus",  "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True, sort_keys = '../JumpToMachineLearning/AppData/python2_lesson06_keys.pkl')
target, features = targetFeatureSplit( data )

# training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

# Linear regression from sklearn

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit (feature_train, target_train)


print("Slope ", reg.coef_)
print("Intercept ", reg.intercept_)

print("Score ", reg.score(feature_train, target_train))
print("Score ", reg.score(feature_test, target_test))


# draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")


# draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass

reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b") 
print("----------------- changeing train and test data-------------")
print("Slope ", reg.coef_)
print("Intercept ", reg.intercept_)

print("Score ", reg.score(feature_train, target_train))
print("Score ", reg.score(feature_test, target_test))
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.savefig("../JumpToMachineLearning/AppData/Plots/regression_on_enron_data.png")
plt.show()

 