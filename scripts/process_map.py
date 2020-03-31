# Copyright (c) 2020, Autonomous Networks Research Group. All rights reserved.
#      contributors: Quynh Nguyen and Bhaskar Krishnamachari
#      Read license file in main directory for more details  
# 
# This script output the geo information for all the regions and generate the corresponding input file for the ArcGIS map

import os
import sys
import pandas as pd 
import matplotlib.pyplot as plt
import json
from opencage.geocoder import OpenCageGeocode
import re
import geopandas as gpd
import seaborn as sns
sns.set(style="darkgrid")




API_KEY = '576004cefa1b43648fd6cd7059ae8196' # get api key from:  https://opencagedata.com
covid_json = 'lacounty_covid.json'
population_json = 'population.json'
population_pdf = 'population.pdf'


def retrieve_all_regions():
    os.chdir('../data/')
    df = pd.read_json(population_json)
    data = df.iloc[:, 0]
    regions = data.apply(lambda x: x.split('--')[0]).values.tolist()
    regions = set(regions)
    return regions

def process_population():
    os.chdir('../map/')
    latlon = pd.read_csv('latlon.csv',header=0)
    os.chdir('../data/')
    df = pd.read_json(population_json)
    for index, row in df.iterrows():
        reg = row[0].split('--')[0]
        idx = latlon.index[latlon['Region'] == reg]
        latlon.loc[idx,'Population'] = int(row[1])
    latlon.to_csv (r'../map/map_population.csv', index = False, header=True)
            
def retrieve_all_regions_covid():
    os.chdir('../data/')
    with open(covid_json, 'r') as j:
        covid = json.loads(j.read())
    regions = set()
    for k,v in covid.items():
        for value in v:
            tmp = value[0].strip()
            try:
                if 'Neighborhood' in tmp or 'Investiga' in tmp or 'Unincorporated' in tmp or 'Communities' in tmp or ' and ' in tmp:
                    continue
                if 'Los Angeles - ' in tmp:
                    tmp2 = tmp.split('-')[1].strip()
                elif 'City of ' in tmp:
                    tmp2 = tmp.split('City of ')[1].strip()
                else:
                    tmp2 = tmp.strip()
                regions.add(tmp2)
            except Exception as e:
                print('Something wrong while parsing')
                print(tmp)
    return regions

def process_covid():
    os.chdir('../map/')
    latlon_covid = pd.read_csv('latlon_covid.csv',header=0)
    os.chdir('../data/')
    with open(covid_json, 'r') as j:
        covid = json.loads(j.read())
    columns = ['Time Stamp','Region', 'Latitude', 'Longitude','Number of cases']
    df = pd.DataFrame(columns = columns)

    
    c = 0
    for k,v in covid.items():
        ts = '03-'+k+'-2020'
        for value in v:
            tmp = value[0].strip()
            try:
                if 'Neighborhood' in tmp or 'Investiga' in tmp or 'Unincorporated' in tmp or 'Communities' in tmp or ' and ' in tmp:
                    continue
                if 'Los Angeles - ' in tmp:
                    reg = tmp.split('-')[1].strip()
                elif 'City of ' in tmp:
                    reg = tmp.split('City of ')[1].strip()
                else:
                    reg = tmp.strip()
                lat = latlon_covid.loc[latlon_covid['Region']==reg,'Latitude'].values[0]
                lon = latlon_covid.loc[latlon_covid['Region']==reg,'Longitude'].values[0]
                cases =  value[1].strip()
                cases = re.sub("[^0-9]", "", cases)
                df.loc[c] = [ts,reg,lat,lon,cases]
                c = c+1
            except Exception as e:
                print('Something wrong while parsing ')
                print(tmp)
    df.to_csv ('../map/Covid-19.csv', index = False, header=True)


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
    latlon.to_csv (r'../map/latlon.csv', index = False, header=True)

def retrieve_gps_covid():
    key = API_KEY  
    geocoder = OpenCageGeocode(key)
    columns = ['Region','Latitude','Longitude']
    latlon = pd.DataFrame(columns=columns)
    regions = retrieve_all_regions_covid()
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
    latlon.to_csv (r'../map/latlon_covid.csv', index = False, header=True)    

def retrieve_covid_date():
    os.chdir('../map/')
    covid = pd.read_csv('Covid-19.csv',header=0)
    if not os.path.exists('covidbydate'):
        os.makedirs('covidbydate')
    date_list = covid['Time Stamp'].unique()
    for d in date_list:
        idx = covid.index[covid['Time Stamp'] == d]
        sub = covid.iloc[idx,:]
        sub = sub.reset_index(drop=True)
        file_name = 'covidbydate/%s.csv'%(d)
        sub.index.name = 'ID'
        sub.to_csv (file_name, index = True, header=True)  

def generate_heatmap_bydate(d):
    os.chdir('../map/')
    regions = gpd.read_file('la/la.shp')
    filename = 'covidbydate/%s.csv'%(d)
    data = pd.read_csv(filename,header=0)
    merged = regions.set_index('name').join(data.set_index('Region'))
    merged = merged.reset_index()
    merged = merged.fillna(0)

    fig, ax = plt.subplots(1, figsize=(40, 20))
    ax.axis('off')
    title = 'Heat Map of Covid-19, Los Angeles County (%s)' %(d)
    ax.set_title(title, fontdict={'fontsize': '40', 'fontweight' : '3'})
    color = 'Oranges'
    vmin, vmax = 0, 231
    sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)
    cbar.ax.tick_params(labelsize=20) 
    merged.plot('Number of cases', cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(40,20))
    # plt.show()
    outfile = 'maps/%s.png'%(d)
    plt.savefig(outfile,bbox_inches='tight')

def generate_heatmap():
    os.chdir('../map/')
    covid = pd.read_csv('Covid-19.csv',header=0)
    date_list = covid['Time Stamp'].unique()
    for d in date_list:
        generate_heatmap_bydate(d)


if __name__ == "__main__":
    # retrieve_gps() # Run this to generate latlon.csv using the API 
    # process_population()
    # retrieve_gps_covid() # Run this to generate latlon_covid.csv using the API 
    # process_covid()
    # retrieve_covid_date()    
    generate_heatmap()