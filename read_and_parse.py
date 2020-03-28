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

def parse_list(xpath):
    ul = tree.xpath(xpath)
    return [child.text for child in ul.getchildren()]

urlcomp = 'http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2285'
headers = {'accept': "application/json", 'accept': "text/csv"}

starting_date=16
data_array={}

for i in range(2271,2285):
    urlcomp="http://publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid="+str(i)
    print(urlcomp)
    # Get JSON Data
    rcomp = requests.get(urlcomp, headers=headers)
    data_array[starting_date]=[]
    soup = BeautifulSoup(rcomp.text,"lxml")
    for ultag in soup.find_all('ul'):
    	for litag in ultag.find_all('li'):
           if "--" in litag.text and "to" not in litag.text and "over" not in litag.text:
               print(litag.text)
               data_array[starting_date].append(litag.text.split("--"))
    starting_date=starting_date+1


#print(data_array)    
# Write to .CSV
#f = open('27mar.csv', "w")
#f.write(rcomp.text)
#f.close()
