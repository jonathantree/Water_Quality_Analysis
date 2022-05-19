from functools import total_ordering
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as soup
import pandas as pd
import requests
import time
import os

start_time = time.time()
#Read in the master data set of zip codes to filter for state on user prompr
us_zip_df = pd.read_csv('../../Resources/Data/simplemaps_uszips_basicv1.80/uszips.csv', dtype={'zip':str})

#Create a list of all of the state ID's to assist user in selecting a state ID
states_list = list(us_zip_df.state_id.unique().ravel())

#Get User input for filtering the us_zip_df
print("Here are the state ID's:")
print(sorted(states_list))
user_state = str(input('Please provide the state id(EX: AL for Alabama): '))

state_zips_df = us_zip_df[us_zip_df.state_id == user_state] 

#Set up a filepath to store data created during the session
# Leaf directory
directory = user_state
 
# Parent Directories
parent_dir = "../../Resources/Data/user_scrape_data/"
 
# Path
path = os.path.join(parent_dir, directory)
 
# Create the directory
# By setting exist_ok as True
# error caused due already
# existing directory can be suppressed
# but other OSError may be raised
# due to other error like
# invalid path name
try:
    os.makedirs(path, exist_ok = True)
    print("Directory '%s' created successfully" %path)
except OSError as error:
    print("Directory '%s' can not be created")

state_path_zip_output = os.path.join(path,'zips.csv')
state_zips_df.to_csv(state_path_zip_output, index=False)

##==================================================================#
## Begin Scraping by finding the utilities serving each zip code
##==================================================================#
#Create a list of the statte's zip codes to iterate through
zip_list = state_zips_df.zip.to_list()

utilities_df = pd.DataFrame()
for zipcode in zip_list:
    page="https://www.ewg.org/tapwater/search-results.php?zip5="+ str(zipcode)+"&searchtype=zip"
    url = requests.get(page)
    
    try:
        table = pd.read_html(url.text)
        data = table[0]
        data['Zip'] = zipcode
        utilities_df = utilities_df.append(data)
        print(utilities_df.values)
    except ValueError:
        print(f'No systems found that match your search for: {zipcode}')
print(utilities_df.shape)
#Output the resulting dataframe to a csv file
print('Writing utilities_with_duplicates file')
state_path_utility_raw_output = os.path.join(path,'utilities_with_duplicates.csv')
utilities_df.to_csv(state_path_utility_raw_output, index=False)

#Get rid of duplicates to result in a unique list of utilities to scrape
utilities_df.drop_duplicates(subset=['Utility name'],inplace=True)

print('Writing utilities file')
#Output the resulting dataframe to a csv file
state_path_utility_output = os.path.join(path,'utilities.csv')
utilities_df.to_csv(state_path_utility_output, index=False)
print(utilities_df.shape)

# PRint Statement to check on progress
print('Done Scraping the Utilities list')
print('Moving on to scraping the contaminant data for each utility')

#Build a list of the utilities to visit and scrape data from EWG
utility_list = utilities_df['Utility name'].to_list()
utility_list

#Begin Scrape
driver = webdriver.Chrome()

contaminant_list = []
for utility in utility_list:
    try:
        driver.get("https://www.ewg.org/tapwater/advanced-search.php")
        time.sleep(2)
        utility_input = driver.find_element(By.XPATH, '/html/body/div[3]/main/div/section[1]/form/input[1]')
        utility_input.clear()
        utility_input.send_keys(utility)
        go_btn = driver.find_element(By.XPATH, '/html/body/div[3]/main/div/section[1]/form/input[3]')
        go_btn.click()
        time.sleep(2)
        utility = driver.find_element(By.XPATH, '/html/body/div[3]/main/figure/table/tbody/tr/td[1]/a')
        utility.click()

        page_source = driver.page_source

        html_soup = soup(page_source, 'html.parser')

        #Get the name of the Water Utility
        Utility = html_soup.find('h1').text
        Utility


        #get the html data we need
        data_box = html_soup.find_all('div', class_='contaminant-name')

        for i in range(len(data_box)):
            data = data_box[i].find_all('span')
            data_measure = []
            d = {
                'Utility' : Utility,
                'Contaminant': '', 
                'Utility Measuremnt':'', 
                'EWG HEALTH GUIDELINE': '',
                'Legal Limit':'' 
            }

            contaminant_name = data_box[i].find('h3')
            d['Contaminant'] = contaminant_name

            for j in range(len(data)):
                measurement = data[j].text
                #print(measurement)
                data_measure.append(measurement)
                #print(data_measure)

            try:
                d['Utility Measuremnt'] = data_measure[data_measure.index('THIS UTILITY')+1]
            except ValueError:
                print("A value error arose")
            except:
                print("Something else went wrong")
            try:
                d['EWG HEALTH GUIDELINE'] = data_measure[data_measure.index('EWG HEALTH GUIDELINE')+1]
            except ValueError:
                print("A value error arose")
            except:
                print("Something else went wrong") 
            try:
                d['Legal Limit'] = data_measure[data_measure.index('LEGAL LIMIT')+1]
            except ValueError:
                print("A value error arose")
            except:
                print("Something else went wrong") 

            contaminant_list.append(d)
    except:
        pass
print ('Done Scraping, moving on to dataset cleaning and file writing')
#Construct a dataframe from the results
scraped_df = pd.DataFrame(contaminant_list)

#Define a function to strip the h3 tages from the contaminants (relict from scraping)
import re
CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

scraped_df = scraped_df.astype({"Contaminant": str})

scraped_df.Contaminant = scraped_df.Contaminant.apply(lambda x: cleanhtml(x))

#Output the resulting dataframe to a csv file
state_path_contaminant_output = os.path.join(path,'contaminants.csv')
scraped_df.to_csv(state_path_contaminant_output, index=False)

finish_time = time.time()

total_time = (finish_time - start_time)/60/60

print(f'The process finished in finished in {total_time} hours')