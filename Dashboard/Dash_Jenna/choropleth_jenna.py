from dash import Dash, dcc, Output, Input  
import dash_bootstrap_components as dbc    
import plotly.express as px
import pandas as pd                       
import sqlite3
from urllib.request import urlopen
import json
import os

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# incorporate data into app
db = r'/Users/jennadodge/uofo-virt-data-pt-12-2021-u-b/Water_Quality_Analysis/Database/database.sqlite3'
# Connect to SQLite database
conn = sqlite3.connect(db)
  
# Create cursor object
cursor = conn.cursor()

#Read in the Data from the DB
census_df = pd.read_sql_query("SELECT * FROM Census_Data" ,conn)
conn.close()
# file_to_read = os.path.join("..","..","Census_Data_Cleaning","zips_to_counties.csv")
zips_to_counties = pd.read_csv("Census_Data_Cleaning/zips_to_counties.csv")
df = pd.merge(zips_to_counties,census_df,how='left',left_on='county_fips',right_on="county_FIPS")

# Build your components
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
mytitle = dcc.Markdown(children='')
mygraph = dcc.Graph(figure={})
myinput = dcc.Input(id='zip_code',
                    type='number',
                    placeholder='Zip Code',
                    value=97701,  # initial value displayed when page first loads
                    )

# Customize your own Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([mytitle], width=6)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mygraph], width=12)
    ]),
    dbc.Row([
        dbc.Col([myinput], width=6)
    ], justify='center'),

], fluid=True)

# Callback allows components to interact
@app.callback(
    [Output(component_id = 'mytitle', component_property = 'children'),
    Output(component_id = 'mygraph', component_property ='figure')],
    [Input(component_id ='zip_code', component_property='value')]
)
def update_graph(zip_code):  # function arguments come from the component property of the Input

    print(zip_code)
    print(type(zip_code))
    dff = df.copy()
    dff = dff[dff['zip']==myinput]
    
    # https://plotly.com/python/choropleth-maps/
    fig = px.choropleth(data_frame = df, 
                        geojson=counties, 
                        locations='county_FIPS', 
                        color="Gini_Index",
                        color_continuous_scale="Viridis",                           
                        scope="usa"
                        )
    fig.update_geos(fitbounds="locations", visible=False)
    return fig, #'# '+zip_code  # returned objects are assigned to the component property of the Output


# Run app
if __name__=='__main__':
    app.run_server(debug=False, port=8044)