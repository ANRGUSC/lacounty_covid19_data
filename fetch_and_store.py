#This script fetches the data from LA County's public health press releases, and 
#creates a dictionary (JSON file) for additional processing

import requests
import json
import csv
from lxml import etree
from lxml import html
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

#headers for GET requests
headers = {'accept': "application/json", 'accept': "text/csv"}

#global variables for storing day and the data_array
starting_date=16 #the data for Covid-19 is available from 16th of March
data_array={} #this dictionary will store all the data

def write_json_to_file():
    global data_array
    out_file = open('lacounty_covid.json','w+')
    json.dump(data_array,out_file)
    return


#the following function retrieves the data from bulleted list
#list_object - Body of the list item (text content)
def parse_list(list_object):
    if ("Hospitalized (Ever)" not in list_object and 
                    "to" not in list_object and  
                    "over" not in list_object and
                    "http" not in list_object and
                    "County" not in list_object and
                    "Long Beach" not in list_object and
                    "Pasadena" not in list_object
                   ):
                    #print(list_object)
                    if "\t" in list_object or "--" in list_object:
                        if "\t" in list_object:
                            out=list_object.split("\t")
                        else:
                            out=list_object.split("--")
                        out[0]=str(out[0]).replace("*","")
                        out[0]=str(out[0]).replace("--","")
                        out[1]=str(out[1]).replace("--","0")
                        out[1]=str(out[1]).replace("0 ","")
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
        if len(data_array[starting_date])==0:
            #print("no data found for day " + str(starting_date))
            for litag in soup.find_all('li'):
                    returned_output=parse_list(litag.text)
                    if returned_output is not None:
                        #print(returned_output)
                        data_array[starting_date].append(returned_output)
        starting_date=starting_date+1                
        return
    


#execution starts here
for press_release_id in range(2268,2286):
    urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid="+str(press_release_id)
    get_data(urlcomp)
#writing dictionary to a file
write_json_to_file()    

print(data_array.keys())    
