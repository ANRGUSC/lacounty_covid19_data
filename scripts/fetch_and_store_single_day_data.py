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

#headers for GET requests
headers = {'accept': "application/json", 'accept': "text/html"}
#headers = {}


#global variables for storing day and the data_array
starting_date=213 #the data for Covid-19 is available from 16th of March
data_array={} #this dictionary will store all the data
case_count={} #dictionary to hold the case count
lacounty_total_case_count={} #count from the press release

#setting up the input file path
script_dir = os.path.dirname(__file__)
rel_path = "../data/lacounty_covid_backup.json"
abs_file_path = os.path.join(script_dir, rel_path)
#setting up the input file path
script_dir = os.path.dirname(__file__)
rel_path_total = "../data/lacounty_total_case_count.json"
abs_file_path_total = os.path.join(script_dir, rel_path_total)




#parsing json
with open(abs_file_path, 'r') as jsonfile:
    data_array=json.load(jsonfile)
print(data_array.keys())
#parsing json
with open(abs_file_path_total, 'r') as jsonfile:
    lacounty_total_case_count=json.load(jsonfile)
print(lacounty_total_case_count.keys())

#del(data_array["153"])

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

#get count from press release
def get_count(key_cursor):
    global data_array,lacounty_total_case_count
    #print(lacounty_total_case_count[key_cursor])
    #print(data_array[key_cursor][0])
    #print(lacounty_total_case_count)
    lacounty_total_case_count[key_cursor]=data_array[key_cursor][0]
    #print(lacounty_total_case_count)
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
                    #"Long Beach" not in list_object and
		    "Hispanic" not in list_object and
		    "White" not in list_object and
		    "Black" not in list_object and
		    "Other" not in list_object and 
		    "Asian" not in list_object and
            "Native Hawaiian/Pacific Islander" not in list_object and
            "American Indian/Alaska Native" not in list_object and 
		    "Under Investigation" not in list_object and
            "Male" not in list_object and
            "Female" not in list_object	
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
                        return out
    else:
        print("+++++++++++++++++++++++++++++++++++")
        #print(list_object)
        final_list=[]
        s1=list_object.split(")")
        if len(s1) > 20:
            for i in s1:
                s2=(i.split("\t(\t"))[0].split("\t")
                if "Under Investigation" not in s2[0]:
                    if "*" in s2[0]:
                         s2[0]=s2[0][:-1]  
                         print(s2[0])
                    if "Unincorporated - Covina (Charter Oak" in s2[0]:
                        final_list.append(["Unincorporated - Covina (Charter Oak)",str("230")])
                    final_list.append(s2)
            print("+++++++++++++++++++++++++++++++++++")
            return final_list
                


        #for i in list_object:
        #    print(i.split("("))
        


#the following function gets the data and store it into a dictionary
#input:
#urlcomp -> URL for the data
def get_data(urlcomp):
    global starting_date,data_array
    rcomp = requests.get(urlcomp, headers=headers)
    print(rcomp.text)
    if "Please see the locations where cases have occurred:" in rcomp.text or "Please see additional information below:" or "Laboratory Confirmed Cases"in rcomp.text:
        print("Case numbers found")
        #print(rcomp.text)
        data_array[starting_date]=[]
        soup = BeautifulSoup(rcomp.text,"lxml")
        html_content = soup.prettify()
        #print(html_content)
        for ultag in soup.find_all('ul'):
            print("-------------------------------------------------")
            print(ultag.text)
            print("-------------------------------------------------")
            for litag in ultag.find_all('li'):
                returned_output=parse_list(litag.text)
                if returned_output is not None:
                    print(returned_output)
                    if len(returned_output) > 20:
                        for i in returned_output:
                            #print(i)
                            data_array[starting_date].append(i)
                    else:
                        data_array[starting_date].append(returned_output)  
        if len(data_array[starting_date])<=2:
            print("no data found for day " + str(starting_date))
            for litag in soup.find_all('li'):
                    returned_output=parse_list(litag.text)
                    if returned_output is not None:
                        #print(returned_output)
                        data_array[starting_date].append(returned_output)
        #starting_date=starting_date+1                
        return


#execution starts here - range entry for the following for loop denotes the press release identifiers
for press_release_id in range(2716,2717):
    print(press_release_id)
    #ignoring a duplicate spanish release
    if press_release_id != 2296 or press_release_id != 2301:
        urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid="+str(press_release_id)
    get_data(urlcomp)
    time.sleep(2)


print(data_array[starting_date])


#counting case
get_count(starting_date)
write_json_to_file("lacounty_total_case_count.json",lacounty_total_case_count)   
#print(lacounty_total_case_count)

#filter to remove duplicate entries from the list based on the string
#remove_element("Pasadena")
remove_element("Los Angeles")

#filter to remove duplicate entries from the list based on the string
#remove_element("Los Angeles County (excl. LB and Pas)")
#remove_element("Los Angeles County (excl. LB and Pas)")
remove_element("Los Angeles")
remove_element("Male")
remove_element("Mmale")
remove_element("Female")
remove_element("Unknown")
remove_element("Under Investigation")
remove_element("  -  Under Investigation")
#remove_element("Los Angeles County (excl. LB and Pas)")


# if str(data_array[starting_date][0][0]).strip()=="Long Beach":
#     print(data_array[starting_date][0])
#     print(data_array[starting_date][0][1])
#     print(data_array[starting_date][1])
#     del(data_array[starting_date][1]) 


# if str(data_array[starting_date][0][0]).strip()=="Pasadena":
#     print(data_array[starting_date][0])
#     print(data_array[starting_date][0][1])
#     print(data_array[starting_date][1])

#print(data_array[starting_date][2])
#del(data_array[starting_date][2]) 
# del(data_array["154"][2])
# del(data_array["154"][2])
# del(data_array["154"][2])
# del(data_array["154"][2])
#del(data_array[starting_date][2])
#del(data_array[starting_date][2])
#del(data_array[starting_date][2])

# for key,value in data_array.items():
#     #print(data_array[key])
#     del(data_array[key][0])
#     #print(data_array[key][0])
#     if "Los Angeles County (excl. LB and Pas)" in data_array[key][0][0]:
#         del(data_array[key][0])
#         #print(data_array[key][0])

#remove_element("Pasadena")        

#print(data_array.keys())
print(data_array[starting_date])

#writing dictionary to a file
write_json_to_file("lacounty_covid.json",data_array)    
