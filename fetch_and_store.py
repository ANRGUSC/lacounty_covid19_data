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

starting_date=16 #the data for Covid-19 is available from 16th of March
data_array={} #this dictionary will store all the data


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
    rcomp = requests.get(urlcomp, headers=headers)
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
                print(returned_output)
                data_array[starting_date].append(returned_output)  
    if len(data_array[starting_date])==0:
        #print("no data found for day " + str(starting_date))
        for litag in soup.find_all('li'):
                returned_output=parse_list(litag.text)
                if returned_output is not None:
                    print(returned_output)
                    data_array[starting_date].append(returned_output)
    return data_array
    


#execution starts here
urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2268"
get_data(urlcomp)
#print(data_array)
starting_date=starting_date+1
for i in range(2271,2276):
    urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid="+str(i)
    get_data(urlcomp)
    #print(data_array)
    starting_date=starting_date+1
urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2277"
get_data(urlcomp)
#print(data_array)
starting_date=starting_date+1
for i in range(2279,2281):
   urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid="+str(i)
   get_data(urlcomp)
   print(data_array)
   starting_date=starting_date+1
urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2282"
get_data(urlcomp)
#print(data_array)
starting_date=starting_date+1
urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2284"
get_data(urlcomp)
#print(data_array)
starting_date=starting_date+1
urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2285"
get_data(urlcomp)
print(data_array)
starting_date=starting_date+1

print(data_array[21])    
