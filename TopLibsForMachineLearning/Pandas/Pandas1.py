import pandas as pd

s = pd.Series([3,4,5,5,9,5,9])
print(s)
print(type(s))

print()

print(s.head())

ufo = pd.read_table("http://bit.ly/uforeports", sep = ",")

print(ufo.head())

print(ufo["State"])

ufo["Location"] = ufo.City + "," + ufo.State

print(ufo.head())
