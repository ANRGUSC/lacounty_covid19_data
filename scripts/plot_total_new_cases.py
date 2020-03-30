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
#print(y_array)
y_array.sort()
new_case_array=[]

for i in range(0,len(y_array)-1):
    new_case_array.append(y_array[i+1]-y_array[i])
print(y_array)
print(new_case_array)
del(x_array[-1])

plt.plot(x_array,new_case_array,marker='o', color='b')
plt.xlabel("Days since March 17, 2020")
plt.ylabel("Cases")
plt.title("LA County Total New Cases")
plt.yscale('log')
plt.show()
