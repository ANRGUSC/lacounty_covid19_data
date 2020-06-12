import requests
import json
import csv
from lxml import etree
from lxml import html
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import os
import time

#global variables for storing day and the data_array
data_array={} #this dictionary will store all the data

#setting up the input file path
script_dir = os.path.dirname(__file__)
rel_path = "../data/population.json"
abs_file_path = os.path.join(script_dir, rel_path)


#parsing json
with open(abs_file_path, "r") as jsonfile:
    data_array=json.load(jsonfile)
print(data_array.keys())
print(len(data_array))

missing=["Covina","Park La Brea","Avalon","Baldwin Park","Bassett","Central","El Monte","El Sereno","Harvard Park","Lake Los Angeles","Rosemead","Rowland Heights","Temple City","Acton","Cerritos","Cloverdale/Cochran","Compton","Downtown","Huntington Park","Koreatown","Mt. Washington","Pasadena","South Pasadena","Wilmington","Agoura Hills","Alhambra","Altadena","Arcadia","Arleta","Azusa","Baldwin Hills","Bell","Bell Gardens","Bellflower","Beverly Hills","Beverlywood","Boyle Heights","Brentwood","Burbank","Calabasas","Canoga Park","Canyon Country","Carson","Castaic","Century City","Chatsworth","Claremont","Crestview","Culver City","Del Rey","Diamond Bar","Downey","Duarte","Eagle Rock","East Hollywood","East Los Angeles","Echo Park","El Segundo","Encino","Exposition Park","Gardena","Glassell Park","Glendale","Glendora","Granada Hills","Hacienda Heights","Hancock Park","Harbor City","Harbor Gateway","Harvard Heights","Hawthorne","Highland Park","Hollywood","Hollywood Hills","Hyde Park","Inglewood","La Canada Flintridge","La Mirada","La Puente","La Verne","Lake Balboa","Lakewood","Lancaster","Lawndale","Leimert Park","Lincoln Heights","Lomita","Los Feliz","Lynwood","Manhattan Beach","Mar Vista","Maywood","Melrose","Miracle Mile","Monrovia","Montebello","Monterey Park","North Hills","North Hollywood","Northridge","Norwalk","Pacoima","Palmdale","Palms","Panorama City","Paramount","Pico Rivera","Playa Vista","Pomona","Porter Ranch","Rancho Palos Verdes","Redondo Beach","Reseda","San Dimas","San Fernando","San Gabriel","San Pedro","Santa Clarita","Santa Monica","Sherman Oaks","Silverlake","South El Monte","South Gate","South Park","South Whittier","Stevenson Ranch","Studio City","Sun Valley","Sunland","Sylmar","Tarzana","Torrance","Tujunga","University Park","Valinda","Valley Glen","Van Nuys","Venice","Vermont Knolls","Vermont Vista","Vernon Central","Walnut","Watts","West Adams","West Covina","West Hills","West Hollywood","West Vernon","Westchester","Westlake","Westwood","Whittier","Willowbrook","Wilshire Center","Winnetka","Woodland Hills"]
print(len(missing))

found=[]


for key in data_array:
	for i in missing:
		if str(i) in key:
			print("Found "+str(i)+" with population of "+str(data_array[key]))
			found.append(str(i))
			missing.remove(str(i))

#print(found)
#print(missing)

