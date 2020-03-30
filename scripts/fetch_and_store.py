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
import os

#headers for GET requests
headers = {'accept': "application/json", 'accept': "text/csv"}

#global variables for storing day and the data_array
starting_date=16 #the data for Covid-19 is available from 16th of March
data_array={} #this dictionary will store all the data
case_count={} #dictionary hold the case count
lacounty_total_case_count={} #count from the press release

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
            if str(data_array[key][l][0]) == input_string:
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

#the following function retrieves the data from bulleted list
#list_object - Body of the list item (text content)
def parse_list(list_object):
    if ("Hospitalized (Ever)" not in list_object and
                    "Death" not in list_object and
                    "Investigated Cases" not in list_object and
                    "0 to 17" not in list_object and
                    "18 to 40" not in list_object and
                    "41 to 65" not in list_object and
                    "over 65" not in list_object and
                    "City of Los Angeles" not in list_object and
                    "http" not in list_object and
                    "Long Beach" not in list_object
                   ):
                    if "\t" in list_object or "--" in list_object:
                        if "\t" in list_object:
                            out=list_object.split("\t")
                        else:
                            out=list_object.split("--")    
                        out[0]=str(out[0]).replace("*","")
                        out[0]=str(out[0]).replace("--","")
                        out[1]=str(out[1]).replace("--","0")
                        out[1]=str(out[1].replace("<",""))
                        out[1]=str(out[1]).lstrip("0")
                        out[1]=str(out[1]).replace("*","")
                        if not out[1]:
                            out[1]="0"
                        #out[1]=str(out[1]).replace("","0")
                        #print(out[1])
                        #out[1]=str(out[1]).replace("0 1","1")
                        #out[1]=str(out[1]).replace("0 2","2")
                        #out[1]=str(out[1]).replace("0 3","3")
                        #out[1]=str(out[1]).replace("0 4","4")
                        #out[1]=str(out[1]).replace("0 5","5")
                        #out[1]=str(out[1]).replace("0 6","6")
                        #out[1]=str(out[1]).replace("0 7","7")
                        #out[1]=str(out[1]).replace("0 8","8")
                        #out[1]=str(out[1]).replace("0 9","9")
                        #out[1]=str(out[1].replace("<",""))
                        return out


#the following function gets the data and store it into a dictionary
#input:
#urlcomp -> URL for the data
def get_data(urlcomp):
    global starting_date,data_array
    rcomp = requests.get(urlcomp, headers=headers)
    if "Please see the locations were cases have occurred:" in rcomp.text:
        print("Case numbers found")
        #print(rcomp.text)
        data_array[starting_date]=[]
        soup = BeautifulSoup(rcomp.text,"lxml")
        html_content = soup.prettify()
        #print(html_content)
        for ultag in soup.find_all('ul'):
            #print(ultag.text)
            for litag in ultag.find_all('li'):
                returned_output=parse_list(litag.text)
                if returned_output is not None:
                    #print(returned_output)
                    data_array[starting_date].append(returned_output)  
        if len(data_array[starting_date])<=3:
            #print("no data found for day " + str(starting_date))
            for litag in soup.find_all('li'):
                    returned_output=parse_list(litag.text)
                    if returned_output is not None:
                        #print(returned_output)
                        data_array[starting_date].append(returned_output)
        starting_date=starting_date+1                
        return
    


#execution starts here
for press_release_id in range(2268,2288):
    urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid="+str(press_release_id)
    get_data(urlcomp)

#filter to remove duplicate entries from the list based on the string
remove_element("Pasadena ")

get_count()
#print(data_array[21])

#counting case
print(lacounty_total_case_count)
write_json_to_file("lacounty_total_case_count.json",lacounty_total_case_count)   

#filter to remove duplicate entries from the list based on the string
remove_element("Los Angeles County (excl. LB and Pas) ")
for key,value in data_array.items():
    del(data_array[key][0])
    print(data_array[key][0])
    if "Los Angeles County (excl. LB and Pas)" in data_array[key][0][0]:
        del(data_array[key][0])
        #print(data_array[key][0])
        
#print(data_array)

#writing dictionary to a file
write_json_to_file("lacounty_covid.json",data_array)    
