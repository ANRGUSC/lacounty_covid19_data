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
import time
import sys

#headers for GET requests
headers = {'accept': "application/json", 'accept': "text/html"}

#global variables for storing day and the data_array
starting_date=256 #the data for Covid-19 is available from 16th of March
press_release_id=2802  #add the id of today's press release
data_array={} #this dictionary will store all the data
case_count={} #dictionary to hold the case count
lacounty_total_case_count={} #count from the press release

#setting up the input file path
script_dir = os.path.dirname(__file__)
rel_path = "../data/lacounty_covid.json"
abs_file_path = os.path.join(script_dir, rel_path)
#setting up the input file path
script_dir = os.path.dirname(__file__)
rel_path_total = "../data/lacounty_total_case_count.json"
abs_file_path_total = os.path.join(script_dir, rel_path_total)
#setting up the file path for death data
script_dir = os.path.dirname(__file__)
out_path_dfile = "../data/lacounty_total_deaths.json" #this line creates a file with log scale in y axis
abs_out_death_count_file = os.path.join(script_dir, out_path_dfile)


#parsing json
with open(abs_file_path, 'r') as jsonfile:
    data_array=json.load(jsonfile)
#print(data_array.keys())
#parsing total case data json
with open(abs_file_path_total, 'r') as jsonfile:
    lacounty_total_case_count=json.load(jsonfile)
#print(lacounty_total_case_count.keys())
#parsing total death data json
with open(abs_out_death_count_file, 'r') as jsonfile:
    lacounty_total_death_count=json.load(jsonfile)
#print(lacounty_total_death_count.keys())

#del data_array["256"]


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
    #for key,value in data_array.items():
    item_to_remove=None
    for l in range(0,len(data_array[starting_date])):
        #print(data_array[key][l])
        if str(data_array[starting_date][l][0].strip()) == input_string.strip():
            #print(data_array[key][l][0])
            item_to_remove=l

    #print(item_to_remove)
    if item_to_remove != None:
       del(data_array[starting_date][item_to_remove])
    return

       
#the following function retrieves the data from bulleted list
#list_object - Body of the list item (text content)
def parse_list(list_object):
    if ("Hospitalized (Ever)" not in list_object and
        "Death" not in list_object and
        "Investigated Cases" not in list_object and
        "0 to 17" not in list_object and
        "18 to 40" not in list_object and
        "41 to 65" not in list_object and
        "over 80" not in list_object and
        "50 to 64" not in list_object and
        "65 to 79" not in list_object and
        "0 to 4" not in list_object and
        "5 to 11" not in list_object and
        "12 to 17" not in list_object and
        "18 to 29" not in list_object and
        "30 to 49" not in list_object and
        "City of Los Angeles" not in list_object and
        "http" not in list_object and
        "Hispanic" not in list_object and
		"White" not in list_object and
		"Black" not in list_object and
		"Other" not in list_object and 
		"Asian" not in list_object and
        "Native Hawaiian/Pacific Islander" not in list_object and
        "American Indian/Alaska Native" not in list_object and 
		"Under Investigation" not in list_object and
        "Male" not in list_object and
        "Female" not in list_object	and
        "LA County residents" not in list_object and
        "to date" not in list_object and
        "ICU"  not in list_object and
        "tested" not in list_object and
        "COVID-19" not in list_object and
        "ICU" not in list_object and
        "http" not in list_object and
        "health" not in list_object and
        "residents" not in list_object and
        "Control" not in list_object and 
        "More than" not in list_object
        ):
            print("Going to look for data after filtering"+str(list_object))
            if "Los Angeles County (excl. LB and Pas)" in list_object:
                removed_spaces=(list_object.replace(" ","")).split("LosAngelesCounty(excl.LBandPas)")
                #print(removed_spaces)
                if not removed_spaces[0]: 
                    if int(removed_spaces[1])>200000:
                        print("total case data found")
                        #print(removed_spaces[1])
                        lacounty_total_case_count[starting_date]=["Los Angeles County (excl. LB and Pas)",str(removed_spaces[1])]
                        write_json_to_file("lacounty_total_case_count.json",lacounty_total_case_count) 
                    elif int(removed_spaces[1])<20000:
                        print("death data found")
                        #print(removed_spaces[1])
                        lacounty_total_death_count[starting_date]=[["Deaths",str(removed_spaces[1])]]
                        print(lacounty_total_death_count[starting_date])
                        write_json_to_file("lacounty_total_deaths.json",lacounty_total_death_count)
                else:
                        print("case data is not found - there may be a spelling mistake in the press release")
                        sys.exit(1)
            elif "Long Beach" in list_object:
                removed_spaces=(list_object.replace(" ","")).split("LongBeach")
                #print(removed_spaces)
                if not removed_spaces[0]: 
                    if int(removed_spaces[1])>1300:
                        print("total case data for long beach found")
                        #print(removed_spaces[1])
                        out=["Long Beach",str(removed_spaces[1])]
                        return out
            elif "Pasadena" in list_object:
                removed_spaces=(list_object.replace(" ","")).split("Pasadena")
                #print(removed_spaces)
                if not removed_spaces[0]: 
                    if int(removed_spaces[1])>2000:
                        print("total case data for pasadena found")
                        #print(removed_spaces[1])
                        out=["Pasadena",str(removed_spaces[1])]
                        return out
            #if "\t" in list_object or "--" in list_object:
            else:
                    if "\t" in list_object:
                        out=list_object.split("\t")
                    else:
                        out=list_object.split("--")  
                        #print(out)  
                    out[0]=str(out[0]).replace("*","")
                    out[0]=str(out[0]).replace("--","")
                    out[1]=str(out[1]).replace("--","0")
                    out[1]=str(out[1].replace("<",""))
                    out[1]=str(out[1]).lstrip("0")
                    out[1]=str(out[1]).replace("*","")
                    if not out[1]:
                        out[1]="0"
                    return out
    else:
        print("++++++++++++++++else+++++++++++")
        print(list_object)
        final_list=[]
        s1=list_object.split(")")
        if len(s1) > 20 and len(s1) < 50:
            for i in s1:
                s2=(i.split("\t(\t"))[0].split("\t")
                if "Under Investigation" not in s2[0]:
                    if "*" in s2[0]:
                         s2[0]=s2[0][:-1]  
                         print(s2[0])
                    print("adding inside else"+str(s2))     
                    final_list.append(s2)
            print("+++++++++++++++++++++++++++++++++++")
            return final_list        


#the following function gets the data and store it into a dictionary
#input:
#urlcomp -> URL for the data
def get_data(urlcomp):
    global starting_date,data_array
    rcomp = requests.get(urlcomp, headers=headers)
    #print(rcomp.text)
    if "Please see the locations where cases have occurred:" in rcomp.text or "Please see additional information below:" or "Laboratory Confirmed Cases"in rcomp.text:
        print("Case numbers found")
        #print(rcomp.text)
        data_array[starting_date]=[]
        soup = BeautifulSoup(rcomp.text,"lxml")
        html_content = soup.prettify()
        #print(html_content)
        for ultag in soup.find_all('ul',text="Please see additional information below:"):
            print("-------------------------------------------------")
            print(ultag.text)
            print("-------------------------------------------------")
            for litag in ultag.find_all('li'):
                print("**********************************************")
                print("litag text"+str(litag.text))
                print("**********************************************")
                returned_output=parse_list(litag.text)
                if returned_output is not None:
                    if len(returned_output) > 20:
                        for i in returned_output:
                            data_array[starting_date].append(i)
                    else:
                        data_array[starting_date].append(returned_output)  
        #sometimes the tags are not formatted correctly on the press release                
        if len(data_array[starting_date])<=3:
            print("no data found for day " + str(starting_date))
            for litag in soup.find_all('li'):
                    returned_output=parse_list(litag.text)
                    if returned_output is not None:
                        data_array[starting_date].append(returned_output)              
        return


#execution starts here - range entry for the following for loop denotes the press release identifiers
urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid="+str(press_release_id)
get_data(urlcomp)
time.sleep(2)


print(data_array[starting_date])


#filter to remove duplicate entries from the list based on the string
remove_element("Los Angeles")

#filter to remove duplicate entries from the list based on the string
remove_element("Los Angeles")
remove_element("Male")
remove_element("Mmale")
remove_element("Female")
remove_element("Unknown")
remove_element("Under Investigation")
remove_element("  -  Under Investigation")

#writing dictionary to a file
write_json_to_file("lacounty_covid.json",data_array)    
