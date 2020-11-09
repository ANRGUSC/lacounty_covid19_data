# Copyright (c) 2020, Autonomous Networks Research Group. All rights reserved.
#      contributors: Gowri Ramachandran, Mehrdad Kiamari, Bhaskar Krishnamachari
#      Read license file in main directory for more details  
# 
# This script fetches the data from LA County's public health press releases, and 
# creates a dictionary (JSON file) for additional processing


import requests
import json
import csv
from lxml import etree
from lxml import html
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time

import pandas as pd
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.patches import Patch
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from scipy.interpolate import interp1d


import os

#setting up the out file path
script_dir = os.path.dirname(__file__)
out_path_log = "../plots/lacounty_total_deaths_log.png" #this line creates a file with log scale in y axis
abs_out_file_path_log_scale = os.path.join(script_dir, out_path_log)
out_path = "../plots/lacounty_total_deaths.png" #this line creates a file with plot in a regular scale
abs_out_file_path = os.path.join(script_dir, out_path)

#setting up the out file path
script_dir = os.path.dirname(__file__)
out_path_nd_log = "../plots/lacounty_total_new_deaths_log.png" #this line creates a file with log scale in y axis
abs_out_file_newdeath_log__scale = os.path.join(script_dir, out_path_nd_log)
out_path_nd = "../plots/lacounty_total_new_deaths.png" #this line creates a file with plot in a regular scale
abs_out_file_newdeath_path = os.path.join(script_dir, out_path_nd)

#setting up the out file path
script_dir = os.path.dirname(__file__)
out_path_dfile = "../data/lacounty_total_deaths.json" #this line creates a file with log scale in y axis
abs_out_death_count_file = os.path.join(script_dir, out_path_dfile)


data_array={} #this dictionary will store all the data

#parsing json
with open(abs_out_death_count_file, 'r') as jsonfile:
    data_array=json.load(jsonfile)
print(data_array.keys())
#del(data_array['38'])
#data_array[starting_date]= [["Deaths", "4945"]]

#  -------------------     The following 2 functions are for plotting -------
def create_dataframe_for_R(new_case_array):  
    c = len(new_case_array)
    data = {}
    data['Deaths'] = new_case_array
    #print(len(data['R']),len(data['Upper']),len(data['Lower']))
    data['Time Stamp'] = pd.date_range(start='2020-03-23', periods=c)    
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
    
    index = result['Deaths'].index.get_level_values('Time Stamp')
    values = result['Deaths'].values
    
    max_value = values.max()
    print("maxxx",max_value)
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               c=cmap(color_mapped(values/(0.4*max_value))),
               edgecolors='k', zorder=2)

    
    extended = pd.date_range(start=pd.Timestamp('2020-03-23'),
                             end=index[-1]+pd.Timedelta(days=1))
    
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0,6.0)
    ax.set_xlim(pd.Timestamp('2020-03-23'), result.index.get_level_values('Time Stamp')[-1]+pd.Timedelta(days=1))
    # #fig.set_facecolor('w')


#write json to a file
def write_json_to_file(filename,array):
    script_dir = os.path.dirname(__file__)
    rel_path = "../data/"+filename
    abs_file_path = os.path.join(script_dir, rel_path)
    out_file = open(abs_file_path,'w+')
    json.dump(array,out_file)
    return

#plotting graphs
#first plot: create a plot for total case count in both linear and log scale
x_array=[]
y_array=[]
i=1;
for key,value in data_array.items():
    x_array.append(int(i))
    print(value)
    y_array.append(int(value[0][1]))
    i=i+1

# print(x_array)
# print(y_array)

y_array.sort()


plt.plot(x_array,y_array,marker='o', color='b')
plt.xlabel("Days since March 22, 2020")
plt.ylabel("Deaths")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(15))
plt.title("LA County Total Deaths")
plt.savefig(abs_out_file_path)
plt.yscale('log')
plt.savefig(abs_out_file_path_log_scale)
time.sleep(5)
plt.close()


#create a plot for total new case in both linear and log scale
new_case_array=[]

for i in range(0,len(y_array)-1):
    new_case_array.append(y_array[i+1]-y_array[i])
#print(y_array)
#print(new_case_array)
del(x_array[-1])


result = create_dataframe_for_R(new_case_array)
fig, ax = plt.subplots(figsize=(600/72,400/72))

state_name = "LA County Total New Deaths"
plot_rt(result, ax,state_name)
print("val")
print(result)

max_value = result['Deaths'].max()
ax.set_title(state_name)
# FIX HERE -------- for fittting the y axis to the largest value in y
ax.set_ylim(0.0,(max_value//100)*100.00+100.00)
# --------------------------------------------------------------------
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
#plt.show()
plt.ylabel("Deaths")
plt.savefig(abs_out_file_newdeath_path)



# plt.plot(x_array,new_case_array,marker='o', color='b')
# plt.xlabel("Days since March 23, 2020")
# plt.ylabel("Deaths")
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
# plt.title("LA County Total New Deaths")
# plt.savefig(abs_out_file_newdeath_path)
# plt.yscale('log')
# plt.savefig(abs_out_file_newdeath_log__scale)
# time.sleep(5)
# plt.close()

# # #writing dictionary to a file
#write_json_to_file(abs_out_death_count_file,data_array)    
