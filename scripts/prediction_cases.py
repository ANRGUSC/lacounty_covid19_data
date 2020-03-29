import json
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

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

def get_population_vec(list_communities):
	with open('population.json') as json_file_pop:
		data = json.load(json_file_pop)
		#print(data)
		dictionary_population_commnunities ={}
		for instance in data:
			dictionary_population_commnunities[instance[0].strip().split('--')[0]] = int(instance[1].strip())
		#print(dictionary_population_commnunities)
		#raise NameError('HiThere')
		output_list = []
		for communiuty_obj in list_communities:
			temp = [val for key,val in dictionary_population_commnunities.items() if communiuty_obj.actual_name == key]
			if temp :
		 		output_list.append(temp.pop())
		if len(output_list) == len(list_communities):
			output = np.asarray(output_list)
		else:
			raise NameError('The name of one of communities has NOT been found!')	 		
		return output
		
# predicition about time T+1 based on data from 0:T
def prediction(A,b, last_I,vec_population):
	# x is the paramter supposed to be estimated: size is c^2
	x, residuals, rank, s = np.linalg.lstsq(A,b, rcond=None)
	# create block A_t
	c = len(vec_population)
	A_t = np.zeros((c,c**2))
	for row in range(c):
		temp_matrix = np.zeros((1,c))
		for col in range(c):
			temp_matrix[0,col] = last_I[col] * (vec_population[row] - last_I[row])
		A_t[row,row*c:row*c+c] = temp_matrix


	return A_t.dot(x) 		
# create matrix A and vector b for linear regression
def create_matrx_A_and_vec_b(matrix_I, vec_population):
	c , T = matrix_I.shape[0] , matrix_I.shape[1]
	b_final = np.zeros((c*(T-1)))
	A_final = np.zeros((c*(T-1),c**2))
	for t in range(T-1):
		b_t = np.zeros((c,))
		A_t = np.zeros((c,c**2))
		for row in range(c):
			b_t[row] = matrix_I[row,t+1]  - matrix_I[row,t]  
			temp_matrix = np.zeros((1,c))
			for col in range(c):
				#print("matrix_I[col,t]",matrix_I[col,t])
				#print("S",vec_population[row] - matrix_I[row,t])
				temp_matrix[0,col] = matrix_I[col,t] * (vec_population[row] - matrix_I[row,t])
				
			A_t[row,row*c:row*c+c] = temp_matrix
		b_final[t*c:t*c+c] = b_t
		A_final[t*c:t*c+c,:] = A_t
	return A_final , b_final 	 			
# create matrix I (infected cases) for selected commnuities
def create_matrix(list_selected_communities):
	# matrix_I  has c rows (communities) and T columns (days) 
	matrix_I =  np.zeros((len(list_selected_communities),len(range(16,list_selected_communities[0].Today_date)) ))
	for i,communiuty_obj in  enumerate(list_selected_communities):
		I_s_this_community_obj =communiuty_obj.plot_info('daily')
		print("row", i, communiuty_obj.name)
		for j,infected_at_this_day in enumerate(I_s_this_community_obj):
			 matrix_I[i,j] = infected_at_this_day
	return matrix_I	 
def main(top_i_comm, type_plot,Today_date):
	dict_county = {} # dictionary of all community objects
	list_communities = [] # list of all community objects
	list_pair = []			
	with open('lacounty_covid.json') as json_file:
		data = json.load(json_file)
		for day in sorted([int(k) for k in data.keys()]):
			if day < Today_date :
				print(day, Today_date )
				for i in range(len(data[str(day)])):
					actual_name_of_community = 	data[str(day)][i][0].strip()
					name_of_community = data[str(day)][i][0].strip().lower().replace(' ','')
					# cleaning city names, removing following prefixes
					prefixex = ['cityof','losangeles-','unincorporated-']
					for word in prefixex:
						name_of_community = name_of_community.replace(word,'') 
					# cleaning confirmed number, e.g. <1 will be 1
					confirmed_cases   = data[str(day)][i][0].strip().lower(),re.sub("[^0-9]", "", data[str(day)][i][1].strip())
					if name_of_community not in dict_county.keys():
						dict_county[name_of_community] = community(name_of_community,actual_name_of_community,Today_date)
						list_communities.append(dict_county[name_of_community ])  
						dict_county[name_of_community].addnumber(day,int(confirmed_cases[1]))
					else:
						dict_county[name_of_community].addnumber(day,int(confirmed_cases[1]))
		# find communiuty with highest 
		for communiuty in dict_county.keys():
			dict_county[communiuty].calculate_total_confirmed_so_far()
	
		#days = list(range(16,Today_date))
		list_selected_communities = []
		days = list(range(1,Today_date-16+1))
		newlist = sorted(list_communities,key=lambda x: x.total_confirmed_so_far, reverse=True)
		for en,communiuty_obj in enumerate(newlist):
			if communiuty_obj.name != '-investigatedcases' and communiuty_obj.name !='-underinvestigation' and top_i_comm > 0:
				# append this city to the list
				list_selected_communities.append(communiuty_obj)
				plt.plot(days, communiuty_obj.plot_info(type_plot),'o-',label = communiuty_obj.actual_name)
				top_i_comm -= 1
		
		matrix_I = create_matrix(list_selected_communities)
		vec_population = get_population_vec(list_selected_communities)
		A,b = create_matrx_A_and_vec_b(matrix_I, vec_population)
		output = prediction(A,b,matrix_I[:,-1],vec_population)
		# plot preditions
		plt.gca().set_prop_cycle(None)
		future_days = list(range(Today_date-16  , Today_date-16 +2))
		for ind_i, communiuty_obj in enumerate(list_selected_communities):
			plt.plot(future_days, [matrix_I[ind_i,-1], matrix_I[ind_i,-1]+output[ind_i]],'o--')
		#print(output)
		plt.legend()
		plt.xlabel('Days since March 16, 2020')
		plt.ylabel('Number of Confirmed Cases')
		plt.grid(True)
		plt.title(type_plot)
		plt.show()		
		
if __name__ == "__main__":
	top_k_community_with_highest_confirmed = 6
	# Display mode: daily or cumulative
	display_mode = 'daily'
	number_of_days_passed_from_16th = 12
	main(top_k_community_with_highest_confirmed,display_mode, 16 + number_of_days_passed_from_16th)		