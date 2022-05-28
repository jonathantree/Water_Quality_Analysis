from dash import Dash, dcc, Output, Input, html  
import dash_bootstrap_components as dbc    
import plotly.express as px
import pandas as pd                       
import sqlite3
from urllib.request import urlopen
import json

#import and clean data
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# Read data
df = pd.read_csv('census_contaminant_priority_by_zip.csv', dtype={"zip":str,"fips":str})
db = r'/Users/jennadodge/uofo-virt-data-pt-12-2021-u-b/Water_Quality_Analysis/Database/database.sqlite3'
conn = sqlite3.connect(db)
# Create cursor object
cursor = conn.cursor()
contaminants_df = pd.read_sql_query("SELECT * FROM all_contaminants",conn)
conn.close()
contaminants_df["Zip"] = contaminants_df["Zip"].astype(str).str[:-2].apply('{:0>5}'.format)  
print(contaminants_df.head())
df_map = df[['Simpson Race Diversity Index','Simpson Ethnic Diversity Index', 'Shannon Race Diversity Index',
       'Shannon Ethnic Diversity Index', 'Gini Index',
       'Number of Contaminants', 'Population Served',
       'Total Contaminant Factor']]

# Build your components
# app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

app = Dash(__name__,
                external_stylesheets=[dbc.themes.LUX],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
                )

# mygraph = dcc.Graph(figure={})
# myhist = dcc.Graph(figure={})
# dropdown = dcc.Dropdown(options=df_map.columns.values,
#                         value='Gini_Index',
#                         clearable=False)
# states_dropdown = dcc.Dropdown(options=[{'label': s, 'value': s} for s in sorted(contaminants_df.State.unique())],
#                         value='VT',
#                         clearable=False)
                        
#--------------------------------------------------------------------------
# Customize your own Layout

app.layout = dbc.Container([
    dbc.Row(html.H1("Water Quality Analysis")),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id = 'dropdown',options=df_map.columns.values,
                        value='Gini_Index',
                        clearable=False), width=6)
    ], justify='center'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='mygraph', figure={}), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='states_dropdown',options=[{'label': s, 'value': s} for s in sorted(contaminants_df.State.unique())],
                        value='VT',
                        clearable=False), width=6)
    ], justify='center'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='myhist', figure={}), width=6)
    ], justify='center'),

], fluid=True)

#-----------------------------------------------------
# Callback allows components to interact
@app.callback(
    Output('mygraph', 'figure'),
    Input('dropdown', 'value')
)

def update_graph(column_name):  # function arguments come from the component property of the Input
    print(column_name)
    print(type(column_name))

    # container = "The column chosen by the user was: {}".format(column_name)

    # https://plotly.com/python/choropleth-maps/
    fig = px.choropleth(
        data_frame = df, 
        # locationmode='USA-states',
        geojson=counties, 
        locations='fips',
        scope='usa', 
        color=column_name,
        color_continuous_scale="Viridis",
        # range_color=(min(column_name.values), max(column_name.values)),                           
        # template='plotly_dark',
        # labels={'Gini_Index':'Gini Index'},
        hover_data=['County',column_name]
    )
    
    
    # fig.update_geos(fitbounds="locations", visible=False)
    # fig.show()
    return fig #, container # returned objects are assigned to the component property of the Output in the order

@app.callback(
    Output('myhist','figure'),
    Input('states_dropdown','value')
)
def update_hist(state_input):
    print(state_input)

    dff = contaminants_df.copy()
    dff = dff[dff['State']==state_input]
    dff = dff.groupby(["Contaminant"]).size().to_frame().sort_values([0], ascending = True).tail(10).reset_index()
    dff = dff.rename(columns={0: 'Count of Contaminant'})

    fig2 = px.histogram(
        data_frame = dff, 
        x = 'Count of Contaminant', 
        y="Contaminant").update_layout(
        title={"text": "Top Ten Contaminants", "x": 0.5}, 
        xaxis_title="Number of Occurences"
    )
    return fig2

# Run app
if __name__=='__main__':
    app.run_server(debug=True, port=8045)