# LA County COVID-19 Data Set and Tools for Data Scientists

This repository contains code (in the form of Python scripts) to obtain and visualize data about confirmed positive cases of COVID-19 in the cities and communities within LA County. It also includes sample data obtained from these scripts as well as sample plots.   

We also post the latest plots every day on the following website: [CoVID-19 Plots for LA County](http://anrg.usc.edu/www/covid19.html).

## Data Source
The Los Angeles Department of Public Health do a press release every day, which contains information about the 
number of CoVID-19 cases in Los Angeles County and its neighborhood. We provide a pointer to some of the press releases that 
were used for scraping the data below:
* [March 16](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2268)
* [March 17](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2271)
* [March 18](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2272)
* [March 19](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2273)
* [March 20](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2274)
* [March 21](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2275)
* [March 22](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2277)
* [March 23](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2279)
* [March 24](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2280)
* [March 25](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2282)
* [March 26](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2284)
* [March 27](http://www.publichealth.lacounty.gov/phcommon/public/media/mediapubhpdetail.cfm?prid=2285)


## Scripts
The scripts folder contains two scripts. The script titled "fetch_and_restore.py" gets the data from the above 
web sites and store them in a JSON file for further processing, visualization, and analytics purposes. For visualization 
purposes, we provide another script (visualization_COVID.py). These scripts have been created to process the press releases 
starting from 16th of March to 27th of March. 

## Data
For individuals interested in the data, you'll find the [data](https://github.com/ANRGUSC/lacounty_covid19_data/tree/master/data) folder to be useful. We provide CSV files of daily Covid-19 cases by community—file named [Covid-19.csv](https://github.com/ANRGUSC/lacounty_covid19_data/blob/master/data/Covid-19.csv). Similarly, this information can be found in JSON files, where the keys represent the "day" in March and the values denote the cases in each community in LA county—files named [lacounty_covid.json](https://github.com/ANRGUSC/lacounty_covid19_data/blob/master/data/lacounty_covid.json) and [lacounty_total_case_count.json](https://github.com/ANRGUSC/lacounty_covid19_data/blob/master/data/lacounty_total_case_count.json). 

## Plots
We have generated plots using the data retrieved from LA county press releases. These plots show the time-series data for confirmed COVID-19 positive cases (daily) and fatalities in the communities and cities within LA County that are showing the most number of cases.

## Questions
For any questions about this data set or tools, please contact Dr. Gowri Sankar Ramachandran (gsramach@usc.edu) or Prof. Bhaskar Krishnamachari (bkrishna@usc.edu). 
