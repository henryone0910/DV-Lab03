# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 02:33:19 2015

@author: nymph
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


############################## Your code for loading and preprocess the data ##
# read data
df = pd.read_csv('household_power_consumption.txt', delimiter=';',\
                parse_dates=['Date'], date_parser=(lambda x : pd.to_datetime(x, format='%d/%m/%Y')))

# filtering the date we will process
df = df.loc[(df.Date == '2007-02-01') | (df.Date == '2007-02-02')]

# preprocess data
df['datetime'] = pd.to_datetime(df.Date.astype('string') + ' ' + df.Time.astype('string'), format='%Y-%m-%d %H:%M:%S')
df = df.iloc[:, 2:]
df[df.columns[:-1]] = df[df.columns[:-1]].astype('float')

############################ Complete the following 4 functions ###############
def plot1():
    bins = np.arange(0, 8, 0.5)
    plt.hist(df['Global_active_power'], bins=bins, color='red', edgecolor='black')

    # Add labels and title
    plt.xlabel('Global Active Power (kilowatts)')
    plt.ylabel('Frequency')
    plt.title('Global Active Power')
    plt.savefig('plot1.png')

def plot2():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(df.datetime, df.Global_active_power, c='black')
    plt.yticks(list(range(0, 7, 2)))
    plt.ylabel('Global Active Power (kilowatts)')
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(day_locator)
    plt.savefig('plot2.png')

def plot3():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(df.datetime, df.Sub_metering_1, c='black', label='Sub_metering_1')
    plt.plot(df.datetime, df.Sub_metering_2, c='red', label='Sub_metering_2')
    plt.plot(df.datetime, df.Sub_metering_3, c='blue', label='Sub_metering_3')
    plt.yticks(list(range(0, 31, 10)))
    plt.ylabel('Energy sub metering')
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(day_locator)
    plt.legend()

    plt.savefig('plot3.png')

def plot4():
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    def set_xticks(ax):
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(day_locator)

    set_xticks_vectorizer = np.vectorize(set_xticks)
    set_xticks_vectorizer(axes)

    axes[0, 0].plot(df.datetime, df.Global_active_power, c='black')
    axes[0, 0].set_yticks(list(range(0, 7, 2)))
    axes[0, 0].set_ylabel('Global Active Power')

    axes[0, 1].plot(df.datetime, df.Voltage, c='black')
    axes[0, 1].set_yticks(list(range(234, 247, 4)))
    axes[0, 1].xaxis.set_major_formatter(date_fmt)
    axes[0, 1].xaxis.set_major_locator(day_locator)
    axes[0, 1].set_ylabel('Voltage')
    axes[0, 1].set_xlabel('datetime')

    axes[1, 0].plot(df.datetime, df.Sub_metering_1, c='black', label='Sub_metering_1')
    axes[1, 0].plot(df.datetime, df.Sub_metering_2, c='red', label='Sub_metering_2')
    axes[1, 0].plot(df.datetime, df.Sub_metering_3, c='blue', label='Sub_metering_3')
    axes[1, 0].set_yticks(list(range(0, 31, 10)))
    axes[1, 0].set_ylabel('Energy sub metering')
    axes[1, 0].legend()

    axes[1, 1].plot(df.datetime, df.Global_reactive_power, c='black')
    axes[1, 1].set_yticks(np.arange(0, 0.6, 0.1))
    axes[1, 1].set_ylabel('Global_reactive_power')
    axes[1, 1].set_xlabel('datetime')

    fig.savefig('plot4.png')

if __name__ == "__main__":
    # define some global variable for ploting
    date_fmt = mdates.DateFormatter('%a')
    day_locator = mdates.DayLocator(interval=1)
    
    # construct plots
    plot1()
    plot2()
    plot3()
    plot4()
