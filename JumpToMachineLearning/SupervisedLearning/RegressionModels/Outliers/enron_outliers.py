import pickle
import sys
import matplotlib.pyplot
sys.path.append("../JumpToMachineLearning/Helpers")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../JumpToMachineLearning/AppData/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.savefig("../JumpToMachineLearning/AppData/Plots/enron_outliers.png")

matplotlib.pyplot.show()


max_salary = 0
max_salary_key = None

for key in data_dict:
	if data_dict[key]["salary"] != 'NaN' and data_dict[key]["salary"] > max_salary:
		max_salary = data_dict[key]["salary"]
		max_salary_key = key

print(max_salary_key)

data_dict.pop(max_salary_key, 0 ) 
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.savefig("../JumpToMachineLearning/AppData/Plots/enron_outliers_cleaned_total.png")

matplotlib.pyplot.show()

# Who made over $5 mil bonus and over $1 mil salary

for key in data_dict:
	if data_dict[key]["salary"] != 'NaN' and data_dict[key]["bonus"] != 'NaN':
		if data_dict[key]["salary"] > 1000000 and data_dict[key]["bonus"] > 5000000:
			print(key)








