#!/usr/bin/env python
# coding: utf-8

# ### Daily R value derived from number of cases reported in LA county
# Mehrdad Kiamari, Bhaskar Krishnamachari - April 2020
# 
# To monitor the severity of any epidemic, it is crucial to look at $R_t$ which is a value representing the effective reproduction number (the number of individuals who are infected per infectious individual at time $t$) of the disease. 
# 
# Regarding $R_t$, the epidemic will exponentially grow among the population when $R_t >> 1$. However, the epidemic sloowly disappear as $R_t<1$. Since restirctions would eventually impactts $R_t$, this measure can guide authorities to take appropriate actions regarding tightening or loosing restrictions for the sake of having economic prosperity and human safety.   
# 
# In this code, we aim at estimating daily R value of COVID-19 in LA county. Our approach is universal and can be utilized for any area. We use SIR model, i.e.
# 
# $$
# \begin{align}
# \frac{dS}{dt} &= -\beta \frac{SI}{N}\\
# \frac{dI}{dt} &= +\beta \frac{SI}{N} - \sigma I\\
# \frac{dR}{dt} &= \sigma I
# \end{align}
# $$
# 
# where $S$, $I$, and $R$ represent the number of Susceptible, Infected, and Recovered people in a population size of $N$. Regarding the parameter $\sigma = \frac{1}{D_i}$, $D_i$ represents the average infectious days.
# 
# As far as $R$ is concerned, it is equal to $\frac{\beta}{\sigma}$. Our idea is to estimate $\beta$ at each time from the above differential equation which involves $\frac{dI}{dt}$, then calculate the corresponding $R$.

# In[1]:


import json
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import numpy as np
from matplotlib.colors import hsv_to_rgb
from itertools import combinations
import pandas as pd
from scipy.optimize import fsolve

from gekko import GEKKO
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

from IPython.display import clear_output
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# Create a dictionary consisting of communities and their population as keys and values, respectively. 

# In[2]:


with open('population.json') as json_file_pop:
	data_population = json.load(json_file_pop)


# Function to check if the communitiy exists in the dictionary for population

# In[3]:


def check_if_community_is_in_dic_pop(community_name):
	with open('population.json') as json_file_pop:
		data_population = json.load(json_file_pop)
		temp = [val for key,val in data_population.items() if community_name == key.strip().split('--')[0]]
		if len(temp)==1:
			#print(community_name,"found")
			return True
		#print(community_name,"NOT found")    
		return False


# ### Side Class and Functions
# 
# Each community is an object with few attributes such as name, number of daily or cumulative cases, etc.
# 

# In[4]:


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
		#print("before")
		#print(self.confirmed)
		index = day - 16
		while index != 0:
			if self.confirmed[index] < self.confirmed[index-1]:
				self.confirmed[index-1] = self.confirmed[index]
			index -= 1
		#print("after")
		#print(self.confirmed)
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

# get population for top selected communities    
def get_population_vec(list_communities):
	with open('population.json') as json_file_pop:
		data = json.load(json_file_pop)
		
		output_list = []
		for communiuty_obj in list_communities:
			temp = [val for key,val in data.items() if communiuty_obj.actual_name == key.strip().split('--')[0]]
			if temp :
		 		output_list.append(int(temp.pop().strip()))
		if len(output_list) == len(list_communities):
			output = np.asarray(output_list)
		else:
			raise NameError('The name of one of communities has NOT been found!')	

		return output

# create matrix for number of infections for top selected communities    
def create_matrix(list_selected_communities,type_plot,til_date):
	# matrix_I  has c rows (communities) and T columns (days) 
	#matrix_I =  np.zeros((len(list_selected_communities),len(range(16,list_selected_communities[0].Today_date)) ))
	matrix_I =  np.zeros((len(list_selected_communities),til_date ))
	for i,communiuty_obj in  enumerate(list_selected_communities):
		I_s_this_community_obj = communiuty_obj.plot_info(type_plot)[:til_date]
		#print("row", i, communiuty_obj.name)
		for j,infected_at_this_day in enumerate(I_s_this_community_obj):
			 matrix_I[i,j] = infected_at_this_day
	return matrix_I	 

# matrix I is supposed to be increasing for each community, so we fix any drop by this function
def fix_matrix_I(matrix_I):
    output = np.zeros_like(matrix_I)
    output[:,0] = matrix_I[:,0]
    r,c = matrix_I.shape[0], matrix_I.shape[1]
    for ind_r in range(r):
        for ind_c in range(1,c):
            if matrix_I[ind_r,ind_c] < matrix_I[ind_r,ind_c-1]:
                output[ind_r,ind_c] = matrix_I[ind_r,ind_c-1]
            else:
                output[ind_r,ind_c] = matrix_I[ind_r,ind_c]
    return output            


# In[5]:


def find_intial_non_zero_val_Infection(ref_matrix_I):
# output : non-zero values for each city    
	vec = np.zeros((ref_matrix_I.shape[0],))
	for city in range(ref_matrix_I.shape[0]):
		for time in range(ref_matrix_I.shape[1]):
			if ref_matrix_I[city,time] != 0:
				vec[city] = ref_matrix_I[city,time]
				break
	return vec			

def function_for_solver(z,*data):
    next_I,curr_I,sigma,N = data
    beta = z
    
    F = 0
    F = ((1/(beta-sigma))*np.log(next_I/((beta-sigma)-beta*next_I/N))) -  ((1/(beta-sigma))*np.log(curr_I/((beta-sigma)-beta*curr_I/N))) - 1.0 # equations

    return F
    
#     
def solve_beta_for_single_time_polynomial(next_I,curr_I,sigma,N,prev_beta):
#      
	if curr_I != 0:
# 		m = GEKKO()             # create GEKKO model
# 		beta = m.Var(value=1.0)      # define new variable, initial value=0
# 		m.Equations([(beta-sigma)*curr_I -  (beta/N)*(curr_I**2) == next_I - curr_I ]) # equations
# 		m.solve(disp=False)     # solve
# 		# not being negative
# 		output = max(beta.value[0],0)
		output = (next_I - curr_I+sigma*curr_I)/(curr_I-(1/N)*curr_I**2)
	else:
		output = prev_beta

	#print(beta.value[0])
	return output 	
def solve_beta_for_single_time_exponential(next_I,curr_I,sigma,N,prev_beta):

	#clear_output(wait=True)    
	#print("curr", curr_I, "next", next_I)
# 	if next_I != 0 and curr_I != 0 and next_I != curr_I:
# 		m = GEKKO()             # create GEKKO model
# 		beta = m.Var(value=1.0)      # define new variable, initial value=0
# 		m.Equations([((1/(beta-sigma))*m.log(next_I/((beta-sigma)-beta*next_I/N))) -  ((1/(beta-sigma))*m.log(curr_I/((beta-sigma)-beta*curr_I/N))) == 1.0]) # equations
# 		m.solve(disp=False)     # solve
# 		output = beta.value[0]
# 	else:
# 		output = solve_beta_for_single_time_polynomial(next_I,curr_I,sigma,N,prev_beta)
##################################
# 	data = (next_I,curr_I,sigma,N)
# 	beta_guess = .2
#	output = fsolve(function_for_solver, beta_guess, args=data)
#################################
	output = solve_beta_for_single_time_polynomial(next_I,curr_I,sigma,N,prev_beta)	
	return output 
def calculating_beta(matrix_I,vec_population,sigma,Today_date, name_top_selected_communities):
	r,c = matrix_I.shape[0] , matrix_I.shape[1]
	matrix_beta = np.zeros((r,c-1))
	R = np.zeros((r,c-1))
	for city in range(r):
		prev_beta = 0
		for time in range(c-1):
			clear_output(wait=True)  
			print("beta for city:",city)
			matrix_beta[city,time] = solve_beta_for_single_time_exponential(matrix_I[city,time+1],matrix_I[city,time],sigma,vec_population[city],prev_beta) 
			prev_beta = matrix_beta[city,time]
			R[city,time] = matrix_beta[city,time] / sigma

	clear_output(wait=True) 
# 	print("DONJE")

	return matrix_beta

def calculate_R_margin_for_single_time(next_I,curr_I,sigma,N):
    # D is ave recovery time in days
    if curr_I != 0: 
        D = 1/sigma;
        std= 4.5
        sigma = 1/(D-std) 
        down = max((next_I- curr_I)/( sigma*(curr_I- (1/N)*curr_I**2) ) + (1/(1-curr_I/N)),0)

        factor = 10
        next_I, curr_I = factor*next_I, factor*curr_I
        sigma = 1/(D+std) 
        up = max((next_I- curr_I)/( sigma*(curr_I- (1/N)*curr_I**2) ) + (1/(1-curr_I/N)),0)
    else:
        down, up = 0, 0
    return down,up

def calculating_R_marigins(matrix_I,vec_population,sigma,Today_date, name_top_selected_communities):
	r,c = matrix_I.shape[0] , matrix_I.shape[1]
#	matrix_beta = np.zeros((r,c-1))
	U = np.zeros((r,c-1))
	D = np.zeros((r,c-1))
	for city in range(r):
		for time in range(c-1):
			clear_output(wait=True)             
			print("Margin for city",city)
			margin = calculate_R_margin_for_single_time(matrix_I[city,time+1],matrix_I[city,time],sigma,vec_population[city]) 
			D[city,time],U[city,time] = margin[0],margin[1]


# 	clear_output(wait=True) 
# 	print("Margin len",D.shape[1],U.shape[1] )

	return D,U
    


# ### Load CSV File

# In[6]:


states1 = pd.read_csv('Covid-19.csv', usecols=[0,1,4],
                     index_col=['Region', 'Time Stamp'],
                     parse_dates=['Time Stamp'],
                     squeeze=True).sort_index()
states = states1.groupby(['Region', 'Time Stamp']).sum()
states.head()
#print(states['Melrose']['2020-03-16'])


# ### Create DataFrame 
# Make DataFrame for R

# In[7]:


def create_dataframe_for_R(ind_city,matrix_beta,sigma,U,D):  
    r,c = matrix_beta.shape[0],matrix_beta.shape[1]
    data={}
    data['R'] = matrix_beta[ind_city,:]/sigma
    data['Upper'] = U[ind_city,:]
    data['Lower'] = D[ind_city,:]
    #print(len(data['R']),len(data['Upper']),len(data['Lower']))
    data['Time Stamp'] = pd.date_range(start='2020-03-16', periods=c)    
    dataset = pd.DataFrame(data)
    dataset.set_index(['Time Stamp'], inplace=True)    
    #print(dataset)
    return dataset    


# ### Plot Func for R

# In[8]:


def plot_rt(result, ax, state_name):
    
    ax.set_title(f"{state_name}")
    
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = result['R'].index.get_level_values('Time Stamp')
    values = result['R'].values
    
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Lower'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
    highfn = interp1d(date2num(index),
                      result['Upper'].values,
                      bounds_error=False,
                      fill_value='extrapolate')
    
    extended = pd.date_range(start=pd.Timestamp('2020-03-16'),
                             end=index[-1]+pd.Timedelta(days=1))
    
    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);
    
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0,10.0)
    ax.set_xlim(pd.Timestamp('2020-03-16'), result.index.get_level_values('Time Stamp')[-1]+pd.Timedelta(days=1))
    #fig.set_facecolor('w')


    


# ### Main Func
# 

# In[10]:


def main(Whole_LAcounty,top_i_comm, type_plot,Today_date,future_day_to_be_predicted,til_date,criteria, sigma,gamma,time_to_show):
	dict_county = {} # dictionary of all community objects
	list_communities = [] # list of all community objects
	list_pair = []			
	with open('lacounty_covid.json') as json_file:
		data = json.load(json_file)
		# record all data by creating community classes and fill out their variables 
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

		# get daily cases of all communities because the cumulative is already obtained
		for communiuty_obj in list_communities:
			for index in range(len(communiuty_obj.confirmed)):
				if index == 0:
					communiuty_obj.confirmed_daily[index] = communiuty_obj.confirmed[index]
				else:
					communiuty_obj.confirmed_daily[index] = communiuty_obj.confirmed[index] - communiuty_obj.confirmed[index-1]	

        
        
        
        
        
		# find communiuty with highest 
		list_selected_communities = []
		days = list(range(1,Today_date-16+1))
		newlist = sorted(list_communities,key=lambda x: x.confirmed[-1], reverse=True)
		for en,communiuty_obj in enumerate(newlist):
			if communiuty_obj.name != '-investigatedcases' and communiuty_obj.name !='-underinvestigation' and top_i_comm > 0:
				# append this city to the list
				if check_if_community_is_in_dic_pop(communiuty_obj.actual_name):
					list_selected_communities.append(communiuty_obj)
					#plt.plot(days, communiuty_obj.plot_info(type_plot),'o-',label = communiuty_obj.actual_name)
					top_i_comm -= 1
		

		#create_csv_file(list_selected_communities)
		# create matrix of I for top communities (highest number of confirmed cases)
		# matrix_I is matrix I for top communities until "til_date" (for training)
		# matrix_I = create_matrix(list_selected_communities, type_plot,til_date)
		# ref_matrix_I is matrix I for top communities for all days 				
		ref_matrix_I = create_matrix(list_selected_communities, type_plot,Today_date-16)

# 		nr,nc = ref_matrix_I.shape[0],ref_matrix_I.shape[1]
# 		for ii in range(nr):
# 			for jj in range(nc):
# 				if jj<nc//2:
# 					ref_matrix_I[ii,jj]= jj
# 				else:
# 					ref_matrix_I[ii,jj]= nc-jj
                    
                    
		# making ref_matrix_I non decreasing
		# ref_matrix_I = fix_matrix_I(ref_matrix_I)


		if Whole_LAcounty == True:        
#################################################################
#             For WHOLE LA county 
#################################################################
			all_communities_available_in_pop_list = []        
			for en,communiuty_obj in enumerate(newlist):
				if communiuty_obj.name != '-investigatedcases' and communiuty_obj.name !='-underinvestigation':
					if check_if_community_is_in_dic_pop(communiuty_obj.actual_name):
						all_communities_available_in_pop_list.append(communiuty_obj)
                    
			vec_pop_all_communities_available = get_population_vec(all_communities_available_in_pop_list) 
			sum_population_all_communities = np.zeros(1)
			sum_population_all_communities[0] = np.sum(vec_pop_all_communities_available)        
			all_communities_matrix_I = create_matrix(all_communities_available_in_pop_list, type_plot,Today_date-16)
			all_communities_matrix_I = fix_matrix_I(all_communities_matrix_I)
			print("s",all_communities_matrix_I.shape)
			summed_over_all_comm_matrix_I = np.reshape(all_communities_matrix_I.sum(axis=0),(1,all_communities_matrix_I.shape[1]))
			print("sum",summed_over_all_comm_matrix_I.shape)
			beta_lacounty = calculating_beta(summed_over_all_comm_matrix_I,sum_population_all_communities,sigma,Today_date, all_communities_available_in_pop_list)
			D_lacounty,U_lacounty = calculating_R_marigins(summed_over_all_comm_matrix_I,sum_population_all_communities,sigma,Today_date, all_communities_available_in_pop_list)
			fig, ax = plt.subplots(figsize=(600/72,400/72))
			state_name = "Daily $R_0$ for LA county"
			result = create_dataframe_for_R(0,beta_lacounty,sigma,U_lacounty,D_lacounty)
			plot_rt(result, ax, state_name)
			#print("val")
			#print(result)
			ax.set_title(f'{state_name}')
			ax.set_ylim(0.1,7.0)
			ax.xaxis.set_major_locator(mdates.WeekdayLocator())
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        
		else:        
			print([obj.actual_name for obj in list_selected_communities])
			vec_population = get_population_vec(list_selected_communities)
			name_top_selected_communities = [obj.actual_name for obj in list_selected_communities]
			#name_top_selected_communities =['Carson', 'South Gate', 'Burbank', 'East Los Angeles', 'Hollywood', 'Downey']
			matrix_beta = calculating_beta(ref_matrix_I,vec_population,sigma,Today_date, name_top_selected_communities)
			#matrix_beta = np.zeros_like(ref_matrix_I)
        
			vec_population = get_population_vec(list_selected_communities)
			# find intial I for each city because the derivation equations are sensirtive to INTIAL values
			# initial_infection_for_SIR = find_intial_non_zero_val_Infection(ref_matrix_I)
			# plot_SIR(matrix_beta,sigma,vec_population, initial_infection_for_SIR, time_to_show,name_top_selected_communities)
			D,U = calculating_R_marigins(ref_matrix_I,vec_population,sigma,Today_date, name_top_selected_communities)
##================== Plotting =====================
			ncols = 3
			nrows = int(np.ceil(len(name_top_selected_communities) / ncols))

			# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))
			fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))
			for i, ax in enumerate(axes.flatten()):
				state_name = name_top_selected_communities[i]
				result = create_dataframe_for_R(i,matrix_beta,sigma,U,D)
				plot_rt(result, axes.flat[i], state_name)
			#import pdb;pdb.set_trace()        
			fig.tight_layout()
			fig.set_facecolor('w')


        
        
if __name__ == "__main__":
	Whole_LAcounty = False
	top_k_community_with_highest_confirmed = 9
	# Display mode: daily or cumulative
	display_mode = 'cumulative'
	number_of_days_passed_from_16th = 78 - 16 + 1	
	number_of_days_passed_from_16th_used_for_prediction =39
	future_day_to_be_predicted = 1
	criteria = 'train'
	# SEIR model 
	# 1-exp(-1/d_I) where d_I is 3-7
	sigma = 1.0/7.5 # 5.2
	gamma = 1.0/(2.3)
	time_to_show = 300 
	""" all_lag_indices represent the lags in the model, it should be a list of increamental numbers (min number is 1), 
	i.g. [1,3] means using times slots information of t-1 and t-3"""
	#all_lag_indices = [1]
	main(Whole_LAcounty,top_k_community_with_highest_confirmed,display_mode, 16 + number_of_days_passed_from_16th,future_day_to_be_predicted,number_of_days_passed_from_16th_used_for_prediction,criteria,sigma,gamma,time_to_show)
        


# In[ ]:





# In[ ]:




