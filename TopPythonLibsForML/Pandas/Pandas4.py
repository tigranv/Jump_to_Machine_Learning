import pandas as pd
import matplotlib.pyplot as plt

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

print(pd.DataFrame({ 'City name': city_names, 'Population': population }))

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
print(california_housing_dataframe.describe())
california_housing_dataframe.hist('housing_median_age')
plt.show()

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(cities['City name'][1])
print(cities[0:2])

print(population.apply(lambda val: val > 1000000))
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
print(cities)

cities.reindex([2, 0, 1])



