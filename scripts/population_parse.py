
import csv
import json
population_dict=[]

with open('sheet.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    print(spamreader)
    for row in spamreader:
        print(row[0])
        population_dict.append([row[0],row[1]])


print(population_dict)
out_file = open('population.json','w+')
json.dump(population_dict, out_file)


