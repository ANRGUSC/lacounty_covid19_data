import csv
import json
import os

#setting up the file path
script_dir = os.path.dirname(__file__)
rel_path = "../data/population_unincorporated_to_main.csv"
abs_file_path = os.path.join(script_dir, rel_path)

rel_path_utoc = "../data/uninc_to_comm.csv"
abs_file_path_utoc = os.path.join(script_dir, rel_path_utoc)

rel_path_ctou = "../data/comm_to_uninc.csv"
abs_file_path_ctou = os.path.join(script_dir, rel_path_ctou)

#setting up the file path
script_dir = os.path.dirname(__file__)
rel_path = "../data/lacounty_covid.json"
abs_file_path_json = os.path.join(script_dir, rel_path)

#community mapped file
out_file="../data/community_mapped_case_data.json"
out_path=os.path.join(script_dir, out_file)



with open(abs_file_path_json, 'r') as jsonfile:
    cases=json.load(jsonfile)


new_case_data_dict={}    

# with open(out_path, 'r') as jsonfile:
#     new_case_data_dict=json.load(jsonfile)




# out_file="../data/population.json"
# out_path=os.path.join(script_dir, out_file)

#this dictionaty will hold the parsed data in JSON format
uninc_to_comm={}
comm_to_uninc={}

#parsing CSV
with open(abs_file_path, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    comm_found=0
    for row in spamreader:
        #print(row)
        updated_index=str("Unincorporated - ") + str(row[0])
        #print(updated_index)
        uninc_to_comm[str(updated_index)]=str(row[1])
        if str(row[1]) in comm_to_uninc:
            comm_to_uninc[str(row[1])].append(str(updated_index)) 
        else:
            comm_to_uninc[str(row[1])]=[]
            comm_to_uninc[str(row[1])].append(str(updated_index))
                   
                
#print(comm_to_uninc["Canyon Country"])



with open(abs_file_path_utoc, 'w') as csv_file:  
   writer = csv.writer(csv_file)
   for key, value in uninc_to_comm.items():
      writer.writerow([key, value])

with open(abs_file_path_ctou, 'w') as csv_file:  
   writer = csv.writer(csv_file)
   for key, value in comm_to_uninc.items():
      writer.writerow([key, value])


#print(cases.keys())
#for k in cases.keys():

total_c=0
total_f=0
total_m_found=0

missing_comm=[]
missing_main_comm=[]

unincorporated_comm_to_delete={}

for e in cases["124"]:
    if str(e[0]).strip() == "El Segundo":
        print(e)


#print(len(cases[k]))

for k in cases:
    for i in range(0, len(cases[k])):
        #print(i)
        comm_found=0
        total_c=total_c+1
        for uninccomm in uninc_to_comm:
            if str(cases[k][i][0].strip()) == str(uninccomm.strip()):
                #print(uninccomm)
                #print("--------------------------")
                comm_exist=0
                #print("top level community is"+str(uninc_to_comm[uninccomm]))
                for e in cases[k]:
                    if (str(e[0].strip()) == str(uninc_to_comm[uninccomm])):
                        #print(e)
                        comm_exist=1
                        #print(uninc_to_comm[uninccomm])
                        #print(e[1])
                        #print(cases[k][i][1])
                        add_cases=int(e[1])+int(cases[k][i][1])
                        #print(add_cases)
                        e=[str(e[0]),str(add_cases)]
                        #print(e)
                if comm_exist == 0:
                    print("added"+str(uninc_to_comm[uninccomm]))
                    cases[k].append([str(uninc_to_comm[uninccomm]),str(cases[k][i][1])])



#key="110"
for key in cases:
    new_list=[]
    for i in cases[key]:
        if "Unincorporated" in str(i[0]).strip():
            #print(i)
            if "Santa Monica Mountains" in str(i[0]).strip() or "Santa Susana Mountains" in str(i[0]).strip():
                new_list.append([i[0],i[1]])
                #print(i)
        else:
            new_list.append([i[0],i[1]])
    new_case_data_dict[key]=new_list

#print(new_list)










# print(total_c)
# print(total_f)

# if total_f == total_c:
#     print("all comm found")
# else:
#     print("not all comm found")
#     #removing duplicates
#     missing_comm = [i for n, i in enumerate(missing_comm) if i not in missing_comm[:n]]        
#     print(missing_comm)


# #print(total_m_found)
# missing_main_comm = [i for n, i in enumerate(missing_main_comm) if i not in missing_main_comm[:n]] 
# print(missing_main_comm)

#print(len(cases[k]))

# for e in cases["124"]:
#     if str(e[0]).strip() == "El Segundo":
#         print(e)

#print(cases[k][300])

#print(unincorporated_comm_to_delete)

#print(new_case_data_dict[key])


#writing dictionary to a JSON file
out_file = open(out_path,'w+')
json.dump(new_case_data_dict, out_file)