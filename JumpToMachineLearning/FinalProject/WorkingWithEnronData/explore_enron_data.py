""" 
    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }
    {features_dict} is a dictionary of features associated with that person.
    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000   
"""

import pickle

enron_data = pickle.load(open("../JumpToMachineLearning/AppData/final_project_dataset.pkl", "rb"))

#for key, value in enron_data.items():
#    print('[{}] : "{}"'.format(key, value))

for key, value in enron_data["SKILLING JEFFREY K"].items():
    print('[{}] - {}'.format(key, value))

# How many data points (people)?
print('Number of persons in dataset - ', len(enron_data))

# For each person, how many features are available?
print('Number of features by person - ', len(enron_data["SKILLING JEFFREY K"]))


# How many POIs are there in the E+F dataset?
countPoi = 0
for key, value in enron_data.items():
    if value["poi"]:
        countPoi += 1
print("Number of POis is : {}".format(countPoi))

# How many POIs are there in total?
poi_reader = open('../JumpToMachineLearning/AppData/poi_names.txt', 'r')
poi_reader.readline() # skip url
poi_reader.readline() # skip blank line
poi_count = 0
for poi in poi_reader:
	poi_count += 1
print("Number of total pois - ", poi_count)


# What is the total value of the stock belonging to James Prentice?
print('Total value of the stock belonging to James Prentice - ', enron_data["PRENTICE JAMES"]["total_stock_value"])

# How many email messages do we have from Wesley Colwell to persons of interest?
print('Number of email messages from Wesley Colwell to persons of interest - ', enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

# What’s the value of stock options exercised by Jeffrey K Skilling?

print('Value of stock options exercised by Jeffrey K Skilling - ', enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

print( "CEO of Enron during most of the time that fraud was - Jeffrey Skilling")
print( "Chairman of the Enron board of directors was - Ken Lay")
print( "CFO (chief financial officer) of Enron during most of the time that fraud was going on - Andrew Fastow")

# Of these three individuals (Lay, Skilling and Fastow), who took home the most money (largest value of â€œtotal_paymentsâ€ feature)?
#LAY KENNETH L, FASTOW ANDREW S, SKILLING JEFFREY K
print('LAY KENNETH L', enron_data['LAY KENNETH L']['total_payments'])
print('FASTOW ANDREW S', enron_data['FASTOW ANDREW S']['total_payments'])
print('SKILLING JEFFREY K', enron_data['SKILLING JEFFREY K']['total_payments'])

most_paid = ''
highest_payment = 0

for key in ('LAY KENNETH L', 'FASTOW ANDREW S', 'SKILLING JEFFREY K'):
	if enron_data[key]['total_payments'] > highest_payment:
		highest_payment = enron_data[key]['total_payments']
		most_paid = key
print('Person who took home the most money  is {}, he took {}$'.format(most_paid, highest_payment))

# How many folks in this dataset have a quantified salary? What about a known email address?
print('Have quantified salary - ', len(dict((key, value) for key, value in enron_data.items() if value["salary"] != 'NaN')))

print('Have known email address? - ', len(dict((key, value) for key, value in enron_data.items() if value["email_address"] != 'NaN')))

# How many people in the E+F dataset (as it currently exists) have NAN for their total payments? What percentage of people in the dataset as a whole is this?
have_nan_for_payments = len(dict((key, value) for key, value in enron_data.items() if value["total_payments"] == 'NaN'))
print('Have NAN for their total payments? - {}, and it is {}%'.format(have_nan_for_payments, have_nan_for_payments/len(enron_data)*100))

# How many POIs in the E+F dataset have NAN for their total payments? What percentage of POIs as a whole is this?
POIs = dict((key,value) for key, value in enron_data.items() if value['poi'] == True)
number_POIs = len(POIs)
no_total_payments = len(dict((key, value) for key, value in POIs.items() if value["total_payments"] == 'NaN'))
print('Have NAN for their total payments? - {}, and it is {}%'.format(no_total_payments, no_total_payments*100/number_POIs))

# If 10 POIs with NaN total_payments were added, what is the new number of people?
print(len(enron_data) + 10)
# What is the new number of people with NaN total_payments?
print(10 + len(dict((key, value) for key, value in enron_data.items() if value["total_payments"] == 'NaN')))

# What is the new number of POIs?
print(10 + len(POIs))

# What percentage have NaN for their total_payments?
print(float(10)/(10 + len(POIs))*100)

# import sys
# sys.path.append("../JumpToMachineLearning/FinalProject/Helperfunctions/")
# import feature_format
#
# feature_list = ["poi", "salary", "bonus"]
# data_array = feature_format.featureFormat( enron_data, feature_list )
# label, features = feature_format.targetFeatureSplit(data_array)
#
# print(data_array)
# print(enron_data['KRAUTZ MICHAEL'])
