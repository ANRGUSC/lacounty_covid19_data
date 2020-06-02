# Copyright (c) 2020, Autonomous Networks Research Group. All rights reserved.
#      contributors: Quynh Nguyen and Bhaskar Krishnamachari
#      Read license file in main directory for more details  
# 
# This script output the geo information for all the regions and generate the corresponding input file for the ArcGIS map

import os
import sys
import pandas as pd 
import matplotlib 
import matplotlib.pyplot as plt
import json
from opencage.geocoder import OpenCageGeocode
import re
import geopandas as gpd
import requests


API_KEY = '576004cefa1b43648fd6cd7059ae8196' # get api key from:  https://opencagedata.com
covid_json = 'lacounty_covid_new_data.json'
population_json = 'population.json'


def retrieve_all_regions():
    os.chdir('../data/')
    with open(population_json, 'r') as j:
        regs = json.loads(j.read())
    regions = set()
    for k,v in regs.items():
        reg = k.split('--')[0].strip()
        regions.add(reg)
    return regions

def process_population():
    os.chdir('../data/')
    latlon = pd.read_csv('latlon.csv',header=0)

    with open(population_json, 'r') as j:
        regs = json.loads(j.read())
    for k,v in regs.items():
        reg = k.split('--')[0].strip()
        idx = latlon.index[latlon['Region'] == reg]
        latlon.loc[idx,'Population'] = int(v)
    latlon.to_csv (r'../data/processed_population.csv', index = False, header=True)
            
def retrieve_all_regions_covid():
    os.chdir('../data/')
    with open(covid_json, 'r') as j:
        covid = json.loads(j.read())
    regions = set()
    for k,v in covid.items():
        for value in v:
            tmp = value[0].strip()
            try:
                if 'Neighborhood' in tmp or 'Investiga' in tmp or 'Communities' in tmp or ' and ' in tmp:
                    continue
                if 'Unincorporated - ' in tmp:
                    tmp2 = tmp.split('-')[1].strip()
                elif 'Los Angeles - ' in tmp:
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
    os.chdir('../data/')
    latlon_covid = pd.read_csv('latlon_covid.csv',header=0)
    with open(covid_json, 'r') as j:
        covid = json.loads(j.read())
    columns = ['Time Stamp','Region', 'Latitude', 'Longitude','Number of cases']
    df = pd.DataFrame(columns = columns)
    c = 0
    for k,v in covid.items():
        if int(k)<32:
            ts = '03-'+k+'-2020'
            print(ts)
        elif int(k) < 62:
            d = int(k)-31
            ts = '04-'+str(d)+'-2020'
            print(ts)
        else:
            print(k)
            d = int(k)-61
            print(d)
            ts = '05-'+str(d)+'-2020'
            print(ts)     
        for value in v:
            tmp = value[0].strip()
            try:
                if 'Neighborhood' in tmp or 'Investiga' in tmp or 'Communities' in tmp or ' and ' in tmp:
                    continue
                if 'Unincorporated' in tmp:
                    reg = tmp.split('-')[1].strip()
                elif 'Los Angeles - ' in tmp:
                    reg = tmp.split('-')[1].strip()
                elif 'City of ' in tmp:
                    reg = tmp.split('City of ')[1].strip()
                else:
                    reg = tmp.strip()
                lat = latlon_covid.loc[latlon_covid['Region']==reg,'Latitude'].values[0]
                lon = latlon_covid.loc[latlon_covid['Region']==reg,'Longitude'].values[0]
                cases =  value[1].strip()
                cases = int(re.sub("[^0-9]", "", cases))
                df.loc[c] = [ts,reg,lat,lon,cases]
                c = c+1
            except Exception as e:
                print('Something wrong while parsing ')
                print(tmp)
    df.to_csv ('../data/Covid-19.csv', index = False, header=True)
    newdf = df.groupby(['Time Stamp','Region', 'Latitude', 'Longitude'])['Number of cases'].sum().reset_index()
    newdf.to_csv ('../data/Covid-19-aggregated.csv', index = False, header=True)

def process_density():
    os.chdir('../data/')
    covid_agg = pd.read_csv('Covid-19-R.csv',header=0)
    population = pd.read_csv('processed_population.csv',header=0)
    columns = ['Time Stamp','Region', 'Latitude', 'Longitude','Density']
    covid_den = pd.DataFrame(columns=columns)
    c = 0
    for index, row in covid_agg.iterrows():
        try:
            idx = population.index[population['Region'] == row['Region']]
            cur_pop = population.loc[idx,'Population'].values[0]
            density = row['R']
            covid_den.loc[c] = [row['Time Stamp'],row['Region'],row['Latitude'],row['Longitude'],density]
            c = c+1
        except Exception as e:
            print('Can not find the community population of '+row['Region'])
    covid_den.to_csv ('/home/gowri/lacounty_covid_data/lacounty_covid19_data/data/R-processing/Covid-19-R-Processed.csv', index = False, header=True)


def retrieve_gps_by_region(reg):
    key = API_KEY  
    geocoder = OpenCageGeocode(key)
    try:
        query = '%s, Los Angeles, USA'%(region)  
        results = geocoder.geocode(query)
        lat= results[0]['geometry']['lat']
        lon = results[0]['geometry']['lng']
        return lat,lon
    except Exception as e:
        print('Can not retrieve region geo info!')
        print(e)    

def retrieve_gps(all_regions):
    key = API_KEY  
    geocoder = OpenCageGeocode(key)
    columns = ['Region','Latitude','Longitude']
    latlon = pd.DataFrame(columns=columns)
    
    c = 0
    for region in all_regions:
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
    latlon.to_csv (r'../data/latlon.csv', index = False, header=True)

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
    latlon.to_csv (r'../data/latlon_covid.csv', index = False, header=True)    

def retrieve_covid_date():
    os.chdir('/home/gowri/lacounty_covid_data/lacounty_covid19_data/data/R-processing/')
    covid = pd.read_csv('Covid-19-R-cleaned.csv',header=0)
    denfold = 'dailycases'
    if not os.path.exists(denfold):
        os.makedirs(denfold)
    date_list = covid['Time Stamp'].unique()
    
    for d in date_list:
        idx = covid.index[covid['Time Stamp'] == d]
        sub = covid.iloc[idx,:]
        sub = sub.reset_index(drop=True)
        file_name = '%s/%s.csv'%(denfold,d)
        sub.index.name = 'ID'
        sub.to_csv (file_name, index = True, header=True)  



def generate_heatmap_bydate(d):
    mapfold='../plots/R-map/'
    if not os.path.exists(mapfold):
        os.makedirs(mapfold)
    os.chdir('/home/gowri/lacounty_covid_data/lacounty_covid19_data/data/R-processing/')
    covid = pd.read_csv('Covid-19-R-cleaned.csv',header=0)
    #print(covid.nlargest(5, ['R']) )
    #max_den = covid['R'].max()
    max_den = 10
    print(max_den)
    regions = gpd.read_file('la.shp')
    filename = 'dailycases/%s.csv'%(d)
    data = pd.read_csv(filename,header=0)
    merged = regions.set_index('name').join(data.set_index('Region'))
    merged = merged.reset_index()
    merged = merged.fillna(0)
    fig, ax = plt.subplots(1, figsize=(40, 20))
    ax.axis('off')
    title = 'Infection Rate Heat Map of Covid-19, Los Angeles County (%s)' %(d)
    ax.set_title(title, fontdict={'fontsize': '40', 'fontweight' : '3'})
    color = 'Oranges'
    vmin, vmax = 0, max_den
    sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)
    cbar.ax.tick_params(labelsize=20) 
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=max_den)
    merged.plot('R', cmap=color,norm=normalize, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(40,20))
    # plt.show()
    outfile = '/home/gowri/lacounty_covid_data/lacounty_covid19_data/plots/R-map/%s.png'%(d)
    plt.savefig(outfile,bbox_inches='tight')
    plt.close()

def generate_heatmap():
    os.chdir('/home/gowri/lacounty_covid_data/lacounty_covid19_data/data/R-processing/')
    covid = pd.read_csv('Covid-19-R-cleaned.csv',header=0)
    date_list = covid['Time Stamp'].unique()
    for d in date_list:
        generate_heatmap_bydate(d)

if __name__ == "__main__":
    # all_regions = retrieve_all_regions()
    # retrieve_gps(all_regions) # Run this to generate latlon.csv using the API 
    # process_population()

    # Run daily
    #retrieve_gps_covid() # Run this to generate latlon_covid.csv using the API 
    #process_covid()
    #process_density()
    #retrieve_covid_date()    
    generate_heatmap()