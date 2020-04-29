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
x_av = movingaverage(new_case_array, 7)
print(x_av)

#plotting data in both linear and log scales
plt.plot(x_array,x_av,marker='o', color='b')
plt.xlabel("Days since March 16, 2020")
plt.ylabel("Deaths")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
plt.title("LA County Total New Cases (7-day moving average)")
#plt.show()
plt.savefig(abs_out_file_path_linear)
plt.yscale('log')
plt.savefig(abs_out_file_path_log)