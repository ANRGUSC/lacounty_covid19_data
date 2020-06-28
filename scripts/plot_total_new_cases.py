import csv
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
    
    max_value = values.max()
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

#setting up the file path
script_dir = os.path.dirname(__file__)
rel_path = "../data/lacounty_total_case_count.json"
abs_file_path = os.path.join(script_dir, rel_path)
#setting up the out file path
script_dir = os.path.dirname(__file__)
out_path_log = "../plots/lacounty_total_confirmed_new_cases_log.png" #this line creates a file with log scale in y axis
abs_out_file_path_log_scale = os.path.join(script_dir, out_path_log)
out_path = "../plots/lacounty_total_confirmed_new_cases.png" #this line creates a file with plot in a regular scale
abs_out_file_path = os.path.join(script_dir, out_path)


#this dictionaty will hold the parsed data in JSON format
population_dict={}

#parsing CSV
with open(abs_file_path, 'r') as jsonfile:
    reader=json.load(jsonfile)

print(reader)
x_array=[]
y_array=[]
i=0;
for key,value in reader.items():
    x_array.append(int(i))
    y_array.append(int(value[1].replace(",","")))
    i=i+1

print(x_array)
#print(y_array)
y_array.sort()
new_case_array=[]

for i in range(0,len(y_array)-1):
    new_case_array.append(y_array[i+1]-y_array[i])
print(y_array)
print(new_case_array)
del(x_array[-1])

result = create_dataframe_for_R(new_case_array)
fig, ax = plt.subplots(figsize=(600/72,400/72))

state_name = "LA County Total New Cases"
plot_rt(result, ax,state_name)
print("val")
print(result)

max_value = result['Cases'].max()
ax.set_title(state_name)
# FIX HERE -------- for fittting the y axis to the largest value in y
ax.set_ylim(0.0,(max_value//100)*100.00+100.00)
# --------------------------------------------------------------------
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

#plt.show()
plt.ylabel("Cases")
plt.savefig(abs_out_file_path)
#ax.set_yscale('log')
#plt.savefig(abs_out_file_path_log_scale)



