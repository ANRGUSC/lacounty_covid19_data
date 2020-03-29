# Copyright (c) 2020, Autonomous Networks Research Group. All rights reserved.
#      contributors: Gowri Ramachandran, Mehrdad Kiamari, Bhaskar Krishnamachari
#      Read license file in main directory for more details  
# 
# This script fetches the data from LA County's public health press releases, and 
# creates a dictionary (JSON file) for additional processing

import os
import sys
import pandas as pd 
import json
from opencage.geocoder import OpenCageGeocode

API_KEY = '576004cefa1b43648fd6cd7059ae8196' # get api key from:  https://opencagedata.com
covid_json = 'lacounty_covid.json'
population_json = 'population.json'
latlon_csv = 'latlon.csv'


def retrieve_all_regions():
    os.chdir('../data/')
    df = pd.read_json(population_json)
    data = df.iloc[:, 0]
    regions = data.apply(lambda x: x.split('--')[0]).values.tolist()
    regions = set(regions)
    return regions

def retrieve_population(latlon):
    df = pd.read_json(population_json)
    for index, row in df.iterrows():
        reg = row[0].split('--')[0]
        idx = latlon.index[latlon['Region'] == reg]
        latlon.loc[idx,'Population'] = int(row[1])
    return latlon
        
def retrieve_covid():
    covid = json.loads(covid_json)
    print(covid)  

def retrieve_gps():
    key = API_KEY  
    geocoder = OpenCageGeocode(key)
    columns = ['Region','Latitude','Longitude']
    latlon = pd.DataFrame(columns=columns)
    regions = retrieve_all_regions()
    c = 0
    for region in regions:
        try:
            print(region)
            query = '%s, Los Angeles, USA'%(region)  
            results = geocoder.geocode(query)
            lat= results[0]['geometry']['lat']
            lon = results[0]['geometry']['lng']
            latlon.loc[c] = [region,lat,lon]
            c = c+1
        except Exception as e:
            print('Can not retrieve region geo info!')
            print(e)
    latlon.to_csv (r'../data/regions.csv', index = False, header=True)

def generate_input_arcgis():
    os.chdir('../data/')
    latlon = pd.read_csv(latlon_csv,header=0)
    latlon = retrieve_population(latlon)
    retrieve_covid()
    # print(latlon)




if __name__ == "__main__":
    # retrieve_gps()
    generate_input_arcgis()
