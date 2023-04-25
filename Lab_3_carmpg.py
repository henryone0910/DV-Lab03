# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 21:53:34 2015

@author: nymph
"""


#################################### Read the data ############################
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt

''' read_csv()
The read_csv() function in pandas package parse an csv data as a DataFrame data structure. What's the endpoint of the data?
The data structure is able to deal with complex table data whose attributes are of all data types. 
Row names, column names in the dataframe can be used to index data.
'''

data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original", delim_whitespace = True, \
 header=None, names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model', 'origin', 'car_name'])

data['mpg']
data.mpg
data.iloc[0,:]

print(data.shape)

################################## Enter your code below ######################

# 1. How many cars and how many attributes are in the data set. 
num_cars, num_attributes = data.shape
print("Number of cars in the dataset:", num_cars)
print("Number of attributes in the dataset:", num_attributes)

""" 2. How many distinct car companies are represented in the data set? What is the name of the car
with the best MPG? What car company produced the most 8-cylinder cars? What are the names
of 3-cylinder cars? """

# How many distinct car companies are represented in the data set?
car_companies = data["car_name"].apply(lambda x: x.split()[0])
num_car_companies = len(car_companies.unique())
print("Number of distinct car companies in the dataset:", num_car_companies)

# What is the name of the car with the best MPG?
idx = data["mpg"].idxmax()
car_name = data.loc[idx, "car_name"]
print("The car with the best MPG is:", car_name)

# What car company produced the most 8-cylinder cars?
eight_cyl_cars = data[data["cylinders"] == 8]
car_companies = eight_cyl_cars["car_name"].apply(lambda x: x.split()[0])
most_common_company = car_companies.value_counts().idxmax()
print("The car company that produced the most 8-cylinder cars is:", most_common_company)

# What are the names of 3-cylinder cars?
three_cyl_cars = data[data["cylinders"] == 3]
car_names = three_cyl_cars["car_name"].tolist()
print("The names of the 3-cylinder cars are:")
for name in car_names:
    print(name)

# 3 What is the range, mean, and standard deviation of each attribute? Pay attention to potential missing values.

stats = data.describe()

print("Range, mean, and standard deviation of each attribute:")
for col in stats.columns:
    attr_range = stats.loc["max", col] - stats.loc["min", col]
    
    print(col)
    print("Range:", attr_range)
    print("Mean:", stats.loc["mean", col])
    print("Standard deviation:", stats.loc["std", col])
    print()

""" 4. Plot histograms for each attribute. Pay attention to the appropriate choice of number of bins.
Write 2-3 sentences summarizing some interesting aspects of the data by looking at the histograms. """

data.hist(bins=20, figsize=(10,10))

plt.suptitle("Histograms of Auto-MPG Dataset Attributes", fontsize=16)
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.show()

''' 5. Plot a scatterplot of weight vs. MPG attributes. What do you conclude about the relationship
between the attributes? What is the correlation coefficient between the 2 attributes?'''

plt.figure()
plt.scatter(data.weight, data.mpg)
plt.xlabel('Weight')
plt.ylabel('Mile per gallon')

plt.show()

print("Correlation coefficient between weight and MPG:", data.weight.corr(data.mpg))

''' 6. Plot a scatterplot of year vs. cylinders attributes. Add a small random noise to the values to make
the scatterplot look nicer. What can you conclude? Do some internet search about the history of car
industry during 70’s that might explain the results.(Hint: data.mpg + np.random.random(len(data.mpg))
will add small random noise)'''

plt.figure()
plt.scatter(data.model, data.cylinders + np.random.random(len(data.cylinders)))
plt.xlabel('Model')
plt.ylabel('Cylinders')

plt.show()

print("Correlation coefficient between model and cylinders:", data.model.corr(data.cylinders))

# 7
plt.figure()
plt.scatter(data.horsepower, data.mpg)
plt.xlabel('Horse Power')
plt.ylabel('Mile per gallon')

plt.figure()
plt.scatter(data.horsepower, data.acceleration)
plt.xlabel('Horse Power')
plt.ylabel('Acceleration')

plt.figure()
plt.scatter(data.horsepower, data.displacement)
plt.xlabel('Horse Power')
plt.ylabel('Displacement')

plt.show()
# 8
# 9

# main fuction
