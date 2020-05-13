import csv
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

#setting up the file path
script_dir = os.path.dirname(__file__)
rel_path = "../data/lacounty_total_case_count.json"
abs_file_path = os.path.join(script_dir, rel_path)
#setting up the out file path
script_dir = os.path.dirname(__file__)
out_path_log = "../plots/lacounty_total_confirmed_cases_log.png" #this line creates a file with log scale in y axis
abs_out_file_path_log_scale = os.path.join(script_dir, out_path_log)
out_path = "../plots/lacounty_total_confirmed_cases.png" #this line creates a file with plot in a regular scale
abs_out_file_path = os.path.join(script_dir, out_path)



#this dictionaty will hold the parsed data in JSON format
population_dict={}

#parsing CSV
with open(abs_file_path, 'r') as jsonfile:
    reader=json.load(jsonfile)

print(reader)
x_array=[]
y_array=[]
i=1;
for key,value in reader.items():
    x_array.append(int(i))
    y_array.append(int(value[1].replace(",","")))
    i=i+1

print(x_array)
print(y_array.sort())


plt.plot(x_array,y_array,marker='o', color='b')
plt.xlabel("Days since March 16, 2020")
plt.ylabel("Cases")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(4))
plt.title("LA County Total Confirmed Cases")
plt.savefig(abs_out_file_path)
plt.yscale('log')
plt.savefig(abs_out_file_path_log_scale)
#plt.yscale('log')
#plt.show()
