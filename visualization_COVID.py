import json
import re
import matplotlib.pyplot as plt
import numpy as np

class community:
	def __init__(self,name,Today_date):
		self.name = name
		self.dic_confirmed = {}
		self.total_confirmed_so_far = 0
		self.Today_date = Today_date
		self.dic_confirmed_cummulative = np.zeros(len(range(16,self.Today_date)),dtype=int)
		
		
	def calculate_total_confirmed_so_far(self):
		for day in self.dic_confirmed.keys():
			self.total_confirmed_so_far += self.dic_confirmed[day]
	def addnumber(self,day, number):
		self.dic_confirmed[day] = number
		# increase cumulative confirmed cases here
		if day == 16:
			self.dic_confirmed_cummulative[day - 16] = number	
		else:
			print(day)
			self.dic_confirmed_cummulative[day - 16] =  self.dic_confirmed_cummulative[day - 16 - 1] +  number	
	def print_stat(self):
		print("name", self.name)
		for day in self.dic_confirmed.keys():
			print(day, self.dic_confirmed[day])
	def plot_info(self,type_plot):
		output = np.zeros(len(range(16,self.Today_date)),dtype=int)
		for index,i in enumerate(list(range(16,self.Today_date))):
			if type_plot == 'daily':
				if i in self.dic_confirmed.keys():
					output[index] =  self.dic_confirmed[i]
				else:
					output[index] = 0
			else:
				output = self.dic_confirmed_cummulative			
		return output	

def main(top_i_comm, type_plot,Today_date):
	dict_county = {}
	list_communities = []
	list_pair = []			
	with open('coviddata.json') as json_file:
		data = json.load(json_file)
		for day in sorted([int(k) for k in data.keys()]):
			for i in range(len(data[str(day)])):
				name_of_community = data[str(day)][i][0].strip().lower().replace(' ','')
				#name_of_community_non_processed = data[str(day)][i][0].strip()
				# cleaning city names, removing prefixes
				prefixex = ['cityof','losangeles-','unincorporated-']
				for word in prefixex:
					name_of_community = name_of_community.replace(word,'') 
				# cleaning confirmed number	
				confirmed_cases   = data[str(day)][i][0].strip().lower(),re.sub("[^0-9]", "", data[str(day)][i][1].strip())
				if name_of_community not in dict_county.keys():
					dict_county[name_of_community] = community(name_of_community,Today_date)
					list_communities.append(dict_county[name_of_community ])  
					dict_county[name_of_community].addnumber(day,int(confirmed_cases[1]))
					#print(i,":",name_of_community,confirmed_cases)	
				else:
					dict_county[name_of_community].addnumber(day,int(confirmed_cases[1]))
					#print("EXIST",i,":",name_of_community,confirmed_cases)
		# find communiuty with highest 
		for communiuty in dict_county.keys():
			dict_county[communiuty].calculate_total_confirmed_so_far()
			#print(dict_county[communiuty].total_confirmed_so_far)

		days = list(range(16,Today_date))
		newlist = sorted(list_communities,key=lambda x: x.total_confirmed_so_far, reverse=True)
		for en,communiuty_obj in enumerate(newlist):
			if communiuty_obj.name != '-investigatedcases' and communiuty_obj.name !='-underinvestigation' and top_i_comm > 0:
				print(en,communiuty_obj.name, communiuty_obj.total_confirmed_so_far)
				print(communiuty_obj.dic_confirmed)
				plt.plot(days, communiuty_obj.plot_info(type_plot),label = communiuty_obj.name)
				print(top_i_comm, communiuty_obj.plot_info(type_plot))
				top_i_comm -= 1
		plt.legend()
		plt.xlabel('Days')
		plt.ylabel('Number of Confirmed Cases')
		plt.show()		
	

if __name__ == "__main__":
	top_k_community_with_highest_confirmed = 6
	# Display mode: daily or cumulative
	display_mode = 'daily'
	number_of_days_passed_from_16th = 12
	main(top_k_community_with_highest_confirmed,display_mode, 16 + number_of_days_passed_from_16th)		
