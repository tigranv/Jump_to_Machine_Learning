import numpy
import random
import matplotlib.pyplot as plt
import sys
sys.path.append("../JumpToMachineLearning/Helpers/")
from prep_data import ageNetWorthData


ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()



from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(ages_train, net_worths_train)

plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")
plt.savefig("../JumpToMachineLearning/AppData/Plots/ages_net_worths_regression.png")
plt.show()

### get Katie's net worth (she's 27)
### sklearn predictions are returned in an array,
### so you'll want to do something like net_worth = predict([27])[0]
### (not exact syntax, the point is that [0] at the end)
km_net_worth = reg.predict(27)

### get the slope, again, you'll get a 2-D array, so stick the [0][0] at the end
slope = reg.coef_

### get the intercepthere you get a 1-D array, so stick [0] on the end to access the info we want

intercept = reg.intercept_


### get the r-squared score on test data
test_score = reg.score(ages_test,net_worths_test)

### get the r-squared score on the training data
training_score = reg.score(ages_train,net_worths_train)

print('"networth" :', km_net_worth)
print('"slope" :', slope)
print('"intercept" :', intercept)
print('"stats on test" :', test_score)
print('"stats on training" :', training_score)



