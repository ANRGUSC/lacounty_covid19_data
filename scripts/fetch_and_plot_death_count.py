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


#headers for GET requests
headers = {'accept': "application/json", 'accept': "text/csv"}

#global variables for storing day and the data_array
starting_date=101 #the data for Covid-19 is available from 16th of March
data_array={} #this dictionary will store all the data

#parsing json
with open(abs_out_death_count_file, 'r') as jsonfile:
    data_array=json.load(jsonfile)
print(data_array.keys())
#del(data_array['38'])


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

#filters duplicate entries
def remove_element(input_string):
    global data_array
    for key,value in data_array.items():
        item_to_remove=None
        for l in range(0,len(value)):
            #print(data_array[key][l])
            if str(data_array[key][l][0].strip()) == input_string:
                print(data_array[key][l][0])
                item_to_remove=l

        #print(item_to_remove)
        if item_to_remove != None:
            del(data_array[key][item_to_remove])
    return

#get count from press release
def get_count():
    global data_array,lacounty_total_case_count
    for key,value in data_array.items():
        #print(key)
        #print(value[0])
        lacounty_total_case_count[key]=value[0]
    return

            
#counts the number of cases
def count_cases():
    global case_count
    #case counting
    cnt=0
    for key,value in data_array.items():
        cnt=0
        for i in value:
            if "Los Angeles County (excl. LB and Pas)" not in i[0]:
                cnt=cnt+int(i[1])
        case_count[key]=cnt    
    return case_count

#counts the number of cases
def count_deaths():
    global data_array,death_count
    #case counting
    for key,value in data_array.items():
        #print(key)
        print(value[1])
        death_count[key]=value[1]
    return

#the following function retrieves the data from bulleted list
#list_object - Body of the list item (text content)
def parse_list(list_object):
    global data_array,cursor
    if ("Los Angeles County (excl. LB and Pas)" in list_object or "Deaths" in list_object):
        #print(list_object)
        out = None
        if "\t" in list_object or "--" in list_object:
            if "\t" in list_object:
                out=list_object.split("\t")
            else:
                out=list_object.split("--") 
        if out == None: 
            print("out is none")  
            out=[]
            out.append("Deaths")
            out.append([int(s) for s in list_object.split() if s.isdigit()][0])   
            print(out)           
        out[0]=str(out[0]).replace("*","")
        out[0]=str(out[0]).replace("--","")
        out[1]=str(out[1]).replace("--","0")
        out[1]=str(out[1].replace("<",""))
        out[1]=str(out[1]).lstrip("0")
        out[1]=str(out[1]).replace("*","")
        #print(out)
        if not out[1]:
            out[1]="0"
        if "Deaths" in str(out[0]):
            data_array[starting_date].append(out) 
        if "Los Angeles County (excl. LB and Pas)" in str(out[0]):
            #filtering death based on the case number value - we assume that 
            #the death numbers below 350. The script has to be updated with a 
            #better parsing when the total death count reaches 250 or so.
            out[1]=out[1].replace(",","")
            if int(out[1]) < 350:
                out[0] = "Deaths"
                data_array[starting_date].append(out)
    return


#the following function gets the data and store it into a dictionary
#input:
#urlcomp -> URL for the data
def get_data(urlcomp):
    global starting_date,data_array
    rcomp = requests.get(urlcomp, headers=headers)
    if "Please see the locations where cases have occurred:" in rcomp.text or "Please see additional information below:" in rcomp.text:
        print("Case numbers found")
        #print(rcomp.text)
        data_array[starting_date]=[]
        soup = BeautifulSoup(rcomp.text,"lxml")
        html_content = soup.prettify()
        #print(html_content)
        for ultag in soup.find_all('ul'):
            #print(ultag.text)
            for litag in ultag.find_all('li'):
                parse_list(litag.text)                
        #starting_date=starting_date+1                
        return
    


#execution starts here - range entry for the following for loop denotes the press release identifiers
for press_release_id in range(2432,2433):
    print(press_release_id)
    #ignoring a duplicate spanish release
    if press_release_id != 2296:
        urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid="+str(press_release_id)
    get_data(urlcomp)

# print(data_array)


#plotting graphs
#first plot: create a plot for total case count in both linear and log scale
x_array=[]
y_array=[]
i=1;
for key,value in data_array.items():
    # print(key)
    # print(value[0][1])
    x_array.append(int(i))
    y_array.append(int(value[0][1]))
    i=i+1

# print(x_array)
# print(y_array)

y_array.sort()


plt.plot(x_array,y_array,marker='o', color='b')
plt.xlabel("Days since March 22, 2020")
plt.ylabel("Deaths")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
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
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
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
write_json_to_file(abs_out_death_count_file,data_array)    
