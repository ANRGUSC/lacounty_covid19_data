#https://ieeexplore.ieee.org/abstract/document/1381948
#http://ceng.usc.edu/~bkrishna/research/papers/LorenzoSecon4cr.pdf

import requests
import json
import csv
from lxml import etree
from lxml import html
from io import StringIO
import re
from bs4 import BeautifulSoup


headers = {'accept': "application/json", 'accept': "text/csv"}

starting_date=16
data_array={}

# Get JSON Data


def get_data(urlcomp,data_array,date_counter):
    rcomp = requests.get(urlcomp, headers=headers)
    data_array[starting_date]=[]
    soup = BeautifulSoup(rcomp.text,"lxml")
    for ultag in soup.find_all('ul'):
        for litag in ultag.find_all('li'):
                if ("Hospitalized (Ever)" not in litag.text and 
                    "to" not in litag.text and  
                    "over" not in litag.text and
                    "http" not in litag.text and
                    "County" not in litag.text and
                    "Long Beach" not in litag.text and
                    "Pasadena" not in litag.text
                   ):
                    #print(litag.text)
                    if "\t" in litag.text or "--" in litag.text:
                        if "\t" in litag.text:
                            out=litag.text.split("\t")
                        else:
                            out=litag.text.split("--")
                        out[0]=str(out[0]).replace("*","")
                        out[0]=str(out[0]).replace("--","")
                        out[1]=str(out[1]).replace("--","0")
                        #print(out)
                        data_array[starting_date].append(out)
    return data_array
    


#execution starts here
urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2268"
get_data(urlcomp,data_array,starting_date)
#print(data_array)
starting_date=starting_date+1
for i in range(2271,2276):
    urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid="+str(i)
    get_data(urlcomp,data_array,starting_date)
    #print(data_array)
    starting_date=starting_date+1
urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2277"
get_data(urlcomp,data_array,starting_date)
#print(data_array)
starting_date=starting_date+1
for i in range(2279,2281):
    urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid="+str(i)
    get_data(urlcomp,data_array,starting_date)
    #print(data_array)
    starting_date=starting_date+1
urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2282"
get_data(urlcomp,data_array,starting_date)
#print(data_array)
starting_date=starting_date+1
urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2284"
get_data(urlcomp,data_array,starting_date)
#print(data_array)
starting_date=starting_date+1
urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2285"
get_data(urlcomp,data_array,starting_date)
print(data_array)
starting_date=starting_date+1

print(data_array.keys())    



