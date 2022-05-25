from dash import Dash, dcc, Output, Input, html  
import dash_bootstrap_components as dbc    
import plotly.express as px
import pandas as pd                       
import sqlite3
from urllib.request import urlopen
import json
import os


#import and clean data
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# # incorporate data into app
# db = r'/Users/jennadodge/uofo-virt-data-pt-12-2021-u-b/Water_Quality_Analysis/Database/database.sqlite3'
# # Connect to SQLite database
# conn = sqlite3.connect(db)
  
# # Create cursor object
# cursor = conn.cursor()

# #Read in the Data from the DB
# census_df = pd.read_sql_query("SELECT * FROM Census_Data" ,conn)
# conn.close()
# # file_to_read = os.path.join("..","..","Census_Data_Cleaning","zips_to_counties.csv")
# zips_to_counties = pd.read_csv("../../Census_Data_Cleaning/zips_to_counties.csv",dtype={"zip": str})
# df['fips']=df['GEOID'].str[-5:]
# df = pd.merge(zips_to_counties,census_df,how='inner',left_on='county_fips',right_on="county_FIPS")
# print(df.head())

# Read data
df = pd.read_csv("census_data_for_graphs.csv", dtype={"zip":str,"fips":str})

# Build your components
app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
mytitle = dcc.Markdown(children='')
mygraph = dcc.Graph(figure={})
zip_input = dcc.Input(#id='zip_code',
                    type='numbers',
                    placeholder='Zip Code',
                    value=97701,
                    debounce = True  # initial value displayed when page first loads
                    )
# dropdown = dcc.Dropdown(#options=['59715','97701'],
#                         options=df["zip"].values,
#                         value='59715',
#                         clearable=False)
#--------------------------------------------------------------------------
# Customize your own Layout
# app.layout = dbc.Container([mytitle, mygraph,dropdown])#, mygraph])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([mytitle], width=6)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mygraph], width=12)
    ]),
    dbc.Row([
        dbc.Col([zip_input], width=6)
    ], justify='center'),

], fluid=True)

#-----------------------------------------------------
# Callback allows components to interact
@app.callback(
    Output(mygraph, 'figure'),
    Output(mytitle,'children'),
    Input(zip_input, 'value')
)
def update_graph(zip_code):  # function arguments come from the component property of the Input
    print(zip_code)
    print(type(zip_code))

    container = "The zip code chosen by the user was: {}".format(zip_code)

    dff = df.copy()
    dff = dff[dff['zip']==zip_code]
    
    # https://plotly.com/python/choropleth-maps/
    fig = px.choropleth(
        data_frame = dff, 
        # locationmode='USA-states',
        geojson=counties, 
        locations='fips',
        scope='usa', 
        color="Gini_Index",
        color_continuous_scale="Viridis",
        range_color=(0.08, 0.70),                           
        # template='plotly_dark',
        labels={'Gini_Index':'Gini Index'},
        hover_data=['County','Gini_Index']
    )
    
    # fig.update_geos(fitbounds="locations", visible=False)
    # fig.show()
    return fig , container # returned objects are assigned to the component property of the Output in the order

# Run app
if __name__=='__main__':
    app.run_server(debug=False, port=8044)