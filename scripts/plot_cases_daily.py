# Copyright (c) 2020, Autonomous Networks Research Group. All rights reserved.
#      contributors: Gowri Ramachandran, Mehrdad Kiamari, Bhaskar Krishnamachari
#      Read license file in main directory for more details  
# 
# This script generates plots of the covid-19 data from LA County 

import json
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as mticker

#setting up the input file path
script_dir = os.path.dirname(__file__)
#rel_path = "../data/lacounty_covid.json"
rel_path = "../data/community_mapped_case_data.json"
abs_file_path = os.path.join(script_dir, rel_path)
#setting up the out file path
script_dir = os.path.dirname(__file__)
out_path_log = "../plots/daily_top6communities_log.png" #this line creates a file with log scale in y axis
abs_out_file_path_log_scale = os.path.join(script_dir, out_path_log)
out_path = "../plots/daily_top6communities.png" #this line creates a file with plot in a regular scale
abs_out_file_path = os.path.join(script_dir, out_path)


class community:
	def __init__(self,name,actual_name,Today_date):
		self.name = name
		self.actual_name = actual_name # for displaying part of figures
		self.dic_confirmed = {}
		self.total_confirmed_so_far = 0
		self.Today_date = Today_date
		self.dic_confirmed_cummulative = np.zeros(len(range(16,self.Today_date)),dtype=int) # for cumulative confirmed cases for each day
		
		
	def calculate_total_confirmed_so_far(self):
		for day in self.dic_confirmed.keys():
			self.total_confirmed_so_far += self.dic_confirmed[day]
	# for adding new entry for each community on every day 		
	def addnumber(self,day, number):
		self.dic_confirmed[day] = number
		# increase cumulative confirmed cases here
		if day == 16:
			self.dic_confirmed_cummulative[day - 16] = number	
		else:
			self.dic_confirmed_cummulative[day - 16] =  self.dic_confirmed_cummulative[day - 16 - 1] +  number	
			print(self.dic_confirmed_cummulative[day - 16])
	# for printing purposes only		
	def print_stat(self):
		print("name", self.name)
		for day in self.dic_confirmed.keys():
			print(day, self.dic_confirmed[day])
	# return the confirmed cases (either daily or cumulative) for each community		
	def plot_info(self,type_plot):
		output = np.zeros(len(range(16,self.Today_date)),dtype=int)
		for index,i in enumerate(list(range(16,self.Today_date))):
			# for daily
			if type_plot == 'daily':
				if i in self.dic_confirmed.keys():
					output[index] =  self.dic_confirmed[i]
				else:
					output[index] = 0
			# for cumulative
			else:
				output = self.dic_confirmed_cummulative			
		return output	

def main(top_i_comm, type_plot,Today_date):
	dict_county = {} # dictionary of all community objects
	list_communities = [] # list of all community objects
	list_pair = []	
        #make sure the json file is in the right directory
	with open(abs_file_path) as json_file:
		data = json.load(json_file)
		print(data.keys())
		for day in sorted([int(k) for k in data.keys()]):
			for i in range(len(data[str(day)])):
				actual_name_of_community = 	data[str(day)][i][0].strip()
				name_of_community = data[str(day)][i][0].strip().lower().replace(' ','')
				# cleaning city names, removing following prefixes
				prefixex = ['cityof','losangeles-','unincorporated-']
				for word in prefixex:
					name_of_community = name_of_community.replace(word,'') 
				# cleaning confirmed number, e.g. <1 will be 1
				confirmed_cases   = data[str(day)][i][0].strip().lower(),re.sub("[^0-9]", "", data[str(day)][i][1].strip())
				#if(name_of_community == "southgate"):
				#	print("++++++++++++++++++++++++++++++++++++++++")
				#	print(name_of_community)
				if name_of_community not in dict_county.keys():
					dict_county[name_of_community] = community(name_of_community,actual_name_of_community,Today_date)
					list_communities.append(dict_county[name_of_community ])  
					dict_county[name_of_community].addnumber(day,int(confirmed_cases[1]))
				else:
					dict_county[name_of_community].addnumber(day,int(confirmed_cases[1]))
		# find communiuty with highest 
		for communiuty in dict_county.keys():
			dict_county[communiuty].calculate_total_confirmed_so_far()
			#print(dict_county[communiuty].total_confirmed_so_far)

		#days = list(range(16,Today_date))
		days = list(range(1,Today_date-16 +1))

		# sort communities in the whole list based on total confirmed cases so far and plot top i communities
		newlist = sorted(list_communities,key=lambda x: x.total_confirmed_so_far, reverse=True)
		for en,communiuty_obj in enumerate(newlist):
			if communiuty_obj.name != '-investigatedcases' and communiuty_obj.name !='-underinvestigation' and communiuty_obj.name !='longbeach' and communiuty_obj.name !='santamonicamountains' and top_i_comm > 0:
				#print(en,communiuty_obj.name, communiuty_obj.total_confirmed_so_far)
				#print(communiuty_obj.dic_confirmed)
				plt.plot(days, communiuty_obj.plot_info(type_plot),'o--',label = communiuty_obj.actual_name)
				#print(top_i_comm, communiuty_obj.plot_info(type_plot))
				top_i_comm -= 1
		plt.legend()
		plt.xlabel('Days since March 16, 2020')
		plt.ylabel('Cases')
		plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(15))
		#plt.yscale('log')
		plt.grid(True)
		plt.title("Cases reported for top 6 communities")
		plt.savefig(abs_out_file_path)
		plt.yscale('log')
		plt.savefig(abs_out_file_path_log_scale)
		#plt.show()		
	

if __name__ == "__main__":
	top_k_community_with_highest_confirmed = 6
	# Display mode: daily or cumulative
	display_mode = 'daily'
	number_of_days_passed_from_16th = 181
	main(top_k_community_with_highest_confirmed,display_mode, 16 + number_of_days_passed_from_16th)