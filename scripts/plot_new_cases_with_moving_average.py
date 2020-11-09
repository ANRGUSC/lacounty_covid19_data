import requests
import json
import csv
from lxml import etree
from lxml import html
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy

import os

import pandas as pd
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.patches import Patch
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from scipy.interpolate import interp1d

#  -------------------     The following 2 functions are for plotting -------
def create_dataframe_for_R(new_case_array):  
    c = len(new_case_array)
    data = {}
    data['Cases'] = new_case_array
    #print(len(data['R']),len(data['Upper']),len(data['Lower']))
    data['Time Stamp'] = pd.date_range(start='2020-03-16', periods=c)    
    dataset = pd.DataFrame(data)
    dataset.set_index(['Time Stamp'], inplace=True)    
    print(dataset)
    return dataset

def plot_rt(result, ax,state_name):
    
    ax.set_title(state_name)
    
    # Colors
    ABOVE  = [1,0,0]   # red
    MIDDLE = [1,.4,.4] # red and gray 1,1,1
    BELOW  = [.2,0,0]  # black 0,0,0
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    print("camp",cmap)
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = result['Cases'].index.get_level_values('Time Stamp')
    values = result['Cases'].values
    
    #max_value = values.max()
    max_value = (values[~np.isnan(values)]).max()
    print("maxxx",max_value)
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               c=cmap(color_mapped(values/(0.4*max_value))),
               edgecolors='k', zorder=2)

    
    extended = pd.date_range(start=pd.Timestamp('2020-03-16'),
                             end=index[-1]+pd.Timedelta(days=1))
    
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0,6.0)
    ax.set_xlim(pd.Timestamp('2020-03-16'), result.index.get_level_values('Time Stamp')[-1]+pd.Timedelta(days=1))
    # #fig.set_facecolor('w')


def movingaverage(interval, window_size):
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')




#setting up the input file path
script_dir = os.path.dirname(__file__)
out_path_dfile = "../data/lacounty_total_case_count.json" #this line creates a file with log scale in y axis
abs_out_death_count_file = os.path.join(script_dir, out_path_dfile)

#setting up the out file path
script_dir = os.path.dirname(__file__)
out_path_linear = "../plots/lacounty_total_new_cases_7_day_average_linear.png" #this line creates a file with log scale in y axis
out_path_log = "../plots/lacounty_total_new_cases_7_day_average_log.png" #this line creates a file with log scale in y axis
abs_out_file_path_linear= os.path.join(script_dir, out_path_linear)
abs_out_file_path_log= os.path.join(script_dir, out_path_log)


data_array={} #this dictionary will store all the data

#parsing json
with open(abs_out_death_count_file, 'r') as jsonfile:
    data_array=json.load(jsonfile)



#Prepating data
x_array=[]
y_array=[]
i=1;
for key,value in data_array.items():
    x_array.append(int(i))
    print(value[1])
    y_array.append(int(value[1].replace(",","")))
    i=i+1

#Sorting data
y_array.sort()

#creating an array for total new cases
new_case_array=[]
for i in range(0,len(y_array)-1):
    new_case_array.append(y_array[i+1]-y_array[i])
del(x_array[-1])

#creating an array of 7-day moving averages
# x_av = movingaverage(new_case_array, 7)
# print(x_av)

# result = create_dataframe_for_R(x_av)
processed_df = create_dataframe_for_R(new_case_array)
result=processed_df.rolling(7).mean()
print(result)

fig, ax = plt.subplots(figsize=(600/72,400/72))

state_name = "LA County Total New Cases (7-day moving average)"
plot_rt(result, ax,state_name)
print("val")
print(result)

max_value = result['Cases'].max()
ax.set_title(state_name)
# FIX HERE -------- for fittting the y axis to the largest value in y
ax.set_ylim(0.0,max_value+10.00)
# --------------------------------------------------------------------
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
#plt.show()


# #plotting data in both linear and log scales
# plt.plot(x_array,x_av,marker='o', color='b')
# plt.xlabel("Days since March 16, 2020")
# plt.ylabel("Deaths")
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
# plt.title("LA County Total New Cases (7-day moving average)")
# #plt.show()
plt.savefig(abs_out_file_path_linear)
# plt.yscale('log')
# plt.savefig(abs_out_file_path_log)
