import pandas as pd

ufo = pd.read_table("http://bit.ly/uforeports", sep = ",")
print(ufo.head())

print('------------------------------')
print()

ufo.drop('Colors Reported', axis=1, inplace=True)
print(ufo.head())

print('------------------------------')
print()