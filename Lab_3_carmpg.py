"""
Created on Tue Jan 27 21:53:34 2015
@author: nymph
"""


#################################### Read the data ############################
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
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

# 3. What is the range, mean, and standard deviation of each attribute? Pay attention to potential missing values.

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

''' 5. Plot a scatterplot of weight vs. MPG attributes. What do you conclude about the relationship
between the attributes? What is the correlation coefficient between the 2 attributes?'''

df = data.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()

x = df.weight
y = df.mpg
plt.figure()
plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)

plt.plot(x,p(x),"r")

plt.title("Relation between weight and MPG", fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Weight', fontsize = 15)
plt.ylabel('Mile per gallon', fontsize = 15)

print("Correlation coefficient between weight and MPG:", data.weight.corr(data.mpg))

''' 6. Plot a scatterplot of year vs. cylinders attributes. Add a small random noise to the values to make
the scatterplot look nicer. What can you conclude? Do some internet search about the history of car
industry during 70’s that might explain the results.(Hint: data.mpg + np.random.random(len(data.mpg))
will add small random noise)'''

plt.figure()

plt.scatter(data.model, data.cylinders + np.random.random(len(data.cylinders)))

plt.title("Cylinders through the years", fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Cylinders', fontsize = 15)

#plt.show()

''' Decrease in the number of cylinders in cars in the 1970s. One of the primary reasons was the oil crisis of 1973, which caused a significant increase in oil
prices (300%) and led to a greater focus on fuel efficiency.'''

# 7. Show 2 more scatterplots that are interesting do you. Discuss what you see. 


x = df.horsepower
y = df.mpg
plt.figure()
plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)

plt.plot(x,p(x),"r")

plt.title("Relation between MPG and HorsePower", fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Horse Power', fontsize = 15)
plt.ylabel('Mile per gallon', fontsize = 15)

x = df.horsepower
y = df.displacement
plt.figure()

plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

plt.plot(x,p(x),"r")

plt.title("Relation between MPG and HorsePower", fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Horse Power', fontsize = 15)
plt.ylabel('Displacement', fontsize = 15)

#plt.show()

# 8 Plot a time series for all the companies that show how many new cars they introduces during
# each year. Do you see some interesting trends?

data_new = data.assign(
    comp = lambda x: x.car_name.str.split().str[0]
)
data_new.model = pd.to_datetime(data_new.model, format = '%y').dt.year

# check spelling

data_new.comp.replace({'chevroelt': 'chevrolet',
                   'maxda':'mazda',
                   'mercedes':'mercedes-benz',
                   'toyouta':'toyota',
                   'vokswagen':'volkswagen',
                    'vw':'volkswagen'
                   }, inplace = True)
# Convert to real company
data_new.comp.replace({'chevrolet': 'General Motors',
                    'dodge': 'General Motors',
                    'buick': 'General Motors',
                    'plymouth': 'General Motors',
                    'pontiac': 'General Motors',
                    'cadillac': 'General Motors',
                    'oldsmobile': 'General Motors',
                    'chevy': 'General Motors',
                    'chrysler': 'General Motors',
                    'mercury': 'Ford',
                    'ford': 'Ford',
                    'capri': 'Ford',
                    'datsun': 'Nissan',     
                    'nissan': 'Nissan',
                    'citroen': 'Stellantis',
                    'opel': 'Stellantis',
                    'fiat': 'Stellantis',
                    'peugeot': 'Stellantis',
                    'audi': 'Volkswagen',
                    'volkswagen': 'Volkswagen',
                    'toyota': 'Toyota',
                    'volvo': 'Volvo',
                    'mazda': 'Mazda',
                    'subaru': 'Subaru',
                    'honda': 'Honda',
                    'mercedes-benz': 'Daimler',
                    'bmw': 'BMW',
                    'saab': 'Saab',
                    'hi': 'Huyndai',
                    'renault': 'Renault',
                    'amc': 'AMC',
                    'triumph': 'British Leyland'
                    }, inplace = True)

seri = data_new.groupby(['comp', 'model']).size().reset_index().rename(columns = {0: 'num'})
seri = seri.pivot(index = 'model', columns = 'comp', values = 'num')
seri = seri.fillna(0)

seri.plot(kind = 'line', figsize = (10, 6), title = 'Number of cars introduced by company during year',fontsize = 15, 
          grid = True, colormap = 'tab20c', marker = 'o', markersize = 5, linewidth = 2)
plt.xlabel('Year',fontsize = 15),
plt.ylabel('Number of cars',fontsize = 15),
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))


plt.figure()
seri['British Leyland'].plot(kind = 'line', figsize = (10, 6),
                     title = 'Number of cars introduced by British Layland during year',fontsize = 15,
                     marker='o', markersize=5, linewidth=2, color = 'orange')
plt.xlabel('Year',fontsize = 15)
plt.ylabel('Number of cars',fontsize = 15)

#plt.show()

# 9. Calculate the pairwise correlation, and draw the heatmap with Matplotlib. Do you see some interesting correlation?

corr = data_new.iloc[:,0:8].corr()

fig, ax = plt.subplots()
plt.pcolor(corr, cmap = 'RdYlGn', linewidths = 0.2)
plt.xticks(np.arange(0.5, len(corr.columns), 1), corr.columns)
plt.yticks(np.arange(0.5, len(corr.columns), 1), corr.columns)
plt.xticks(rotation = 45);
plt.colorbar()
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(i + 0.5, j + 0.5, '%.2f' % corr.iloc[i,j], ha = 'center', va = 'center')
plt.title('Correlation between features')
plt.show()



