import json
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
import os
import matplotlib.ticker as mticker

#setting up the file path
script_dir = os.path.dirname(__file__)
#covid_data_path = "../data/lacounty_covid.json"
covid_data_path = "../data/community_mapped_case_data.json"
covid_data_file_path = os.path.join(script_dir, covid_data_path)
#setting up the file path
script_dir = os.path.dirname(__file__)
#population_path = "../data/population.json"
population_path = "../data/population_whole_county.json"
pop_data_file_path = os.path.join(script_dir, population_path)
#setting up the out file path
script_dir = os.path.dirname(__file__)
out_path = "../plots/lacounty_case_density.png" 
abs_out_file_path = os.path.join(script_dir, out_path)


class community:
	def __init__(self,name,actual_name,Today_date):
		self.name = name
		self.Today_date = Today_date
		self.actual_name = actual_name # for displaying part of figures
		# cumulative total
		self.confirmed = np.zeros(len(range(16,self.Today_date)),dtype=int)
		self.confirmed_daily = np.zeros(len(range(16,self.Today_date)),dtype=int)
	# for adding new entry for each community on every day 	
	def check_validity_new_entry(self,day):
		index = day - 16
		if index == 0:
			return True
		else:
			if self.confirmed[index] >= self.confirmed[index-1]:
				return True
			return False		
	def update_confirmed_cases(self,day):
		index = day - 16
		while index != 0:
			if self.confirmed[index] < self.confirmed[index-1]:
				self.confirmed[index-1] = self.confirmed[index]
			index -= 1

	def addnumber(self,day, number):
		index = day - 16
		self.confirmed[index] = number
		status_validity_of_entry = self.check_validity_new_entry(day)
		if not status_validity_of_entry:
			self.update_confirmed_cases(day)

	# return the confirmed cases (either daily or cumulative) for each community		
	def plot_info(self,type_plot):
		output = np.zeros(len(range(16,self.Today_date)),dtype=int)
		for index,i in enumerate(list(range(16,self.Today_date))):
			# for daily
			if type_plot == 'daily':
				# if i in self.dic_confirmed.keys():
				# 	output[index] =  self.dic_confirmed[i]
				# else:
				# 	output[index] = 0
				output = self.confirmed_daily
			# for cumulative
			else:
				output = self.confirmed
		return output	

def get_population_vec(list_communities):
	with open(pop_data_file_path) as json_file_pop:
		data = json.load(json_file_pop)
		
		output_list = []
		for communiuty_obj in list_communities:
			print(communiuty_obj.actual_name)
			prefixex = ['cityof','losangeles-','unincorporated-','Los Angeles-','Los Angeles -',' Los Angeles-']
			for word in prefixex:
				communiuty_obj.actual_name = (communiuty_obj.actual_name.replace(word,'')).strip()
			print(communiuty_obj.actual_name)
			temp = [val for key,val in data.items() if communiuty_obj.actual_name == key.strip().split('--')[0]]
			print(temp)
			if temp :
		 		output_list.append(int(temp.pop().strip()))
		if len(output_list) == len(list_communities):
			output = np.asarray(output_list)
		else:
			raise NameError('The name of one of communities has NOT been found!')	

		return output
		
def main(top_i_comm, type_plot,Today_date):
	dict_county = {} # dictionary of all community objects
	list_communities = [] # list of all community objects
	list_pair = []			
	with open(covid_data_file_path) as json_file:
		data = json.load(json_file)
		for day in sorted([int(k) for k in data.keys()]):
			if day < Today_date :
				#print(day, Today_date )
				for i in range(len(data[str(day)])):
					actual_name_of_community = 	data[str(day)][i][0].strip()
					name_of_community = data[str(day)][i][0].strip().lower().replace(' ','')
					# cleaning city names, removing following prefixes
					prefixex = ['cityof','losangeles-','unincorporated-','Los Angeles-','Los Angeles -',' Los Angeles-']
					for word in prefixex:
						name_of_community = name_of_community.replace(word,'') 
						#print(name_of_community)
					# cleaning confirmed number, e.g. <1 will be 1
					confirmed_cases   = data[str(day)][i][0].strip().lower(),re.sub("[^0-9]", "", data[str(day)][i][1].strip())
					if(name_of_community != "longbeach"):
						print(name_of_community)
						if name_of_community not in dict_county.keys():
							dict_county[name_of_community] = community(name_of_community,actual_name_of_community,Today_date)
							list_communities.append(dict_county[name_of_community ])  
							dict_county[name_of_community].addnumber(day,int(confirmed_cases[1]))
						else:
							dict_county[name_of_community].addnumber(day,int(confirmed_cases[1]))

		# get daily cases of all communities because the cumulative is already obtained
		for communiuty_obj in list_communities:
			for index in range(len(communiuty_obj.confirmed)):
				if index == 0:
					communiuty_obj.confirmed_daily[index] = communiuty_obj.confirmed[index]
				else:
					communiuty_obj.confirmed_daily[index] = communiuty_obj.confirmed[index] - communiuty_obj.confirmed[index-1]	
		#import pdb; pdb.set_trace()				
		# find communiuty with highest 
		# for communiuty in dict_county.keys():
		# 	dict_county[communiuty].calculate_total_confirmed_so_far()
	
		#days = list(range(16,Today_date))
		list_selected_communities = []
		days = list(range(1,Today_date-16+1))
		newlist = sorted(list_communities,key=lambda x: x.confirmed[-1], reverse=True)
		for en,communiuty_obj in enumerate(newlist):
			if communiuty_obj.name != '-investigatedcases' and communiuty_obj.name !='-underinvestigation' and top_i_comm > 0:
				# append this city to the list
				#print(communiuty_obj)
				list_selected_communities.append(communiuty_obj)
				plt.plot(days, communiuty_obj.plot_info(type_plot)/(get_population_vec([communiuty_obj])[0]*1.0),'o-',label = communiuty_obj.actual_name)
				top_i_comm -= 1
		plt.legend()
		plt.xlabel('Days since March 16, 2020')
		plt.ylabel('Cases/population (Case Density)')
		plt.grid(True)
		plt.tight_layout()
		plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))
		# plt.yscale('log')
		plt.title('Case Density for Top 6 Communities')
		#plt.savefig('testfig.png',dpi=300, bbox_inches = "tight")
		plt.savefig(abs_out_file_path,bbox_inches = "tight")
		#plt.show()		
		
if __name__ == "__main__":
	top_k_community_with_highest_confirmed = 6
	# Display mode: daily or cumulative
	display_mode = 'cumulative'
	number_of_days_passed_from_16th = 161
	main(top_k_community_with_highest_confirmed,display_mode, 16 + number_of_days_passed_from_16th)