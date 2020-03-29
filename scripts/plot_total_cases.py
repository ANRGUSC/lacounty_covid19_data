import csv
import json
import os
import matplotlib.pyplot as plt

#setting up the file path
script_dir = os.path.dirname(__file__)
rel_path = "../data/lacounty_total_case_count.json"
abs_file_path = os.path.join(script_dir, rel_path)


#this dictionaty will hold the parsed data in JSON format
population_dict={}

#parsing CSV
with open(abs_file_path, 'r') as jsonfile:
    reader=json.load(jsonfile)

print(reader)
x_array=[]
y_array=[]
i=0;
for key,value in reader.items():
    x_array.append(int(i))
    y_array.append(int(value[1]))
    i=i+1

print(x_array)
print(y_array.sort())


plt.plot(x_array,y_array,marker='o', color='b')
plt.xlabel("Days since March 16, 2020")
plt.ylabel("Cases")
plt.title("LA County Total Confirmed Cases")
plt.yscale('log')
plt.show()
