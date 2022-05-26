from functools import total_ordering
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as soup
import pandas as pd
import requests
import time
import os
import numpy as np
from dash import Dash, dcc, Output, Input, html  
import dash_bootstrap_components as dbc    
import plotly.express as px
from urllib.request import urlopen
import json

app = Dash(__name__,
                external_stylesheets=[dbc.themes.SANDSTONE],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
                )
# Read data
df = pd.read_csv("census_data_for_graphs.csv", dtype={"zip":str,"fips":str})
#import and clean data
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# Build your components
app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
# mytitle = dcc.Markdown(children='')
# # mygraph = dcc.Graph(figure={})
# total_cont = dcc.Markdown(children='')
# zip_input = dcc.Input(#id='zip_code',
#                     type='numbers',
#                     placeholder='Zip Code',
#                     # value='',
#                     debounce = True  # initial value displayed when page first loads
#                     )
# # submit_button = html.Button(n_clicks=0,children='Submit')
# user_zip = input('Enter a zip code and press enter ')
# print(f'Searching for Utilities serving  {user_zip} and the surrrounding cities for your county')

app.layout = dbc.Container([
     dbc.Row(
        dbc.Col(html.H1("Water Quality Analysis",
                        className='text-center text-primary mb-4'),
                width=12)
    ),
    dbc.Row([
        dbc.Col([dcc.Markdown(id='confirmation',children='')], width=12)
    ], justify='center'),
    dbc.Row([
        dbc.Col([dcc.Markdown(id='total_cont',children='')], width=12)
    ]),
    dbc.Row([
        dbc.Col(html.H3("Enter a Zip Code:")),
        dbc.Col([dcc.Input(id='zip_input',
                    type='numbers',
                    placeholder='Zip Code',
                    # value='',
                    debounce = True  # initial value displayed when page first loads
                    )], width=8)
    ], justify='center'),

], fluid=True)

#-----------------------------------------------------
# Callback allows components to interact
@app.callback(
    Output("confirmation",'children'),
    Input('zip_input', 'value')
)
def confirm_zip(zip_code):
    
    container = "The zip code chosen by the user was: {}".format(zip_code)

    return container

@app.callback(
    Output('total_cont', 'children'),
    Input('zip_input', 'value')
)
def update_graph(zip_code):  # function arguments come from the component property of the Input
    print(zip_code)
    print(type(zip_code))

    # container = "The zip code chosen by the user was: {}".format(zip_code)

    page="https://www.ewg.org/tapwater/search-results.php?zip5="+ str(zip_code)+"&searchtype=zip"
    url = requests.get(page)
    table = pd.read_html(url.text)
    utilities_df = pd.concat([table[0], table[1]], ignore_index=True)
    #Build a list of the utilities to visit and scrape data from EWG
    utility_list = utilities_df['Utility name'].to_list()

    start_time = time.time()

    #Begin Scrape
    driver = webdriver.Chrome()

    contaminant_list = []
    for utility in utility_list:
        try:
            driver.get("https://www.ewg.org/tapwater/advanced-search.php")
            time.sleep(1)
            utility_input = driver.find_element(By.XPATH, '/html/body/div[3]/main/div/section[1]/form/input[1]')
            utility_input.clear()
            utility_input.send_keys(utility)
            go_btn = driver.find_element(By.XPATH, '/html/body/div[3]/main/div/section[1]/form/input[3]')
            go_btn.click()
            time.sleep(1)
            utility = driver.find_element(By.XPATH, '/html/body/div[3]/main/figure/table/tbody/tr/td[1]/a')
            utility.click()

            page_source = driver.page_source

            html_soup = soup(page_source, 'html.parser')

            #Get the name of the Water Utility
            Utility = html_soup.find('h1').text

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
    print('Done Scraping, moving on to dataset cleaning and file writing')
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

    # #Output the resulting dataframe to a csv file
    # state_path_contaminant_output = os.path.join(path,'contaminants.csv')
    # scraped_df.to_csv(state_path_contaminant_output, index=False)

    finish_time = time.time()

    total_time = (finish_time - start_time)/60

    print(f'The process finished in finished in {total_time} minutes')

    df = scraped_df.copy()
    df =df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna()
    df3 = df.copy()
    #Split of the contaminant measurements and units and people served
    df3['Units'] = df3['Utility Measuremnt'].apply(lambda x: x.split()[-1])
    df3['Utility Measuremnt'] = df3['Utility Measuremnt'].apply(lambda x: x.split()[0])
    df3['EWG HEALTH GUIDELINE'] = df3['EWG HEALTH GUIDELINE'].apply(lambda x: x.split()[0])
    df3['Legal Limit']=df3['Legal Limit'].apply(lambda x: '0 units' if pd.isnull(x) else x)
    df3['Legal Limit'] = df3['Legal Limit'].apply(lambda x: x.split()[0])
    df3.replace(',','', regex=True, inplace=True)
    #Change the datatype of the measurements and EWG Guidlines to numeric values
    df3['Utility Measuremnt'] = pd.to_numeric(df3['Utility Measuremnt'])
    df3['EWG HEALTH GUIDELINE'] = pd.to_numeric(df3['EWG HEALTH GUIDELINE'])
    #Define the Contaminant Factor (how many times larger is the utility measurement than EWG guideline)
    df3['Contaminant_Factor'] = df3['Utility Measuremnt']/df3['EWG HEALTH GUIDELINE']
    utilities_grouped_df1 = df3.groupby('Utility').count()['Contaminant_Factor']
    utilities_grouped_df2 = df3.groupby('Utility').sum()['Contaminant_Factor']
    Sum_ContaminantFactor = utilities_grouped_df2.sum()
    Avg_Contaminant_Factor = utilities_grouped_df2.mean()
    Num_Contaminants = utilities_grouped_df1.sum()
    #Output to be joined with census data for racial, ethnic, and income indices
    data = {
        'Zip' : zip_code,
        'Sum_ContaminantFactor': Sum_ContaminantFactor, 
        'Avg_Contaminant_Factor':Avg_Contaminant_Factor, 
        'Num_Contaminants' : Num_Contaminants
    }

    df_final = pd.DataFrame(data, index=[0])
    cont_num = df_final.loc[0,'Num_Contaminants']
    total_contaminants = "The total contaminants found in your area is: {}".format(cont_num)

    return total_contaminants

    # # https://plotly.com/python/choropleth-maps/
    # fig = px.choropleth(
    #     data_frame = df_final, 
    #     # locationmode='USA-states',
    #     geojson=counties, 
    #     locations='fips',
    #     scope='usa', 
    #     color="Gini_Index",
    #     color_continuous_scale="Viridis",
    #     range_color=(0.08, 0.70),                           
    #     # template='plotly_dark',
    #     labels={'Gini_Index':'Gini Index'},
    #     hover_data=['County','Gini_Index']
    # )
    
    # fig.update_geos(fitbounds="locations", visible=False)
    # fig.show()
    # return fig , container # returned objects are assigned to the component property of the Output in the order

# Run app
if __name__=='__main__':
    app.run_server(debug=False, port=8047)