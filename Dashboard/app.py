from dash import Dash, dcc, Output, Input, html  
import dash_bootstrap_components as dbc    
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd                       
import sqlite3
from urllib.request import urlopen
import json
import dash_daq as daq

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

app = Dash(__name__,
                external_stylesheets=[dbc.themes.LUX],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
                )
                        
#--------------------------------------------------------------------------
# Customize the Layout

app.layout = dbc.Container([
    dbc.Row(html.H1("Water Quality Analysis")),
    dbc.Row([
        dbc.Col(html.H4("Please select a value"), width=6),
        dbc.Col(dcc.Dropdown(id = 'dropdown',options=df_map.columns.values,
                        value='Gini Index',
                        clearable=False), width=6)
    ], justify='center'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='mygraph', figure={}), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.H4("Please Select a State"), width = 6),
        dbc.Col(dcc.Dropdown(id='states_dropdown',options=[{'label': s, 'value': s} for s in sorted(contaminants_df.State.unique())],
                        value='VT',
                        clearable=False), width=6)
    ], justify='center'),
    dbc.Row([
        dbc.Col(html.H4("Zip Code:")),                
        dbc.Col(dcc.Input(id='zip_code',
                    type='number',
                    # placeholder='',
                    value=97701,
                    debounce = True  # initial value displayed when page first loads
                    ),width=6),
    ], justify='center'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='myhist', figure={}), width=6),
        # dbc.Col(daq.Gauge(id='gauge',
        #     label='Priority',
        #     scale={'start':0,'interval':1,'labelInterval':1},
        #     value = dff.Priority,
        #     min=0,
        #     max=3,
        #      ), width=6)
        dbc.Col(dcc.Graph(id='gauge',figure={}),width=6)
    ], justify='center'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='scatter', figure={}), width=12)
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

@app.callback(
    Output('scatter','figure'),
    Input('states_dropdown','value')
)

def update_scatter(state_input):

    c_df = contaminants_df.copy()
    c_df2 = c_df[c_df.State == state_input]
    top_c_df =c_df2.groupby(by=["Contaminant"]).sum().sort_values(by=['Contaminant_Factor'], ascending=False)[['People_served', 'Contaminant_Factor']]
    top15_c_df = top_c_df.head(15)
    top15_c_df = top15_c_df.reset_index()

    fig3 = px.scatter(
        data_frame = top15_c_df, 
        x="Contaminant", 
        y="People_served",
        size="Contaminant_Factor", 
        color="Contaminant",
        hover_name="Contaminant", 
        size_max=60
    )
    return fig3

@app.callback(
    Output('gauge','figure'),
    Input('zip_code','value')
)

def update_gauge(zip):
    zip = str(zip)
    dff = df.copy()
    dff = dff[dff['zip']==zip]

    fig4 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value=dff.Priority.values[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text':'Priority','font':{'size':24}},
        gauge = {
            'axis':{'range':[-0.5,2.5]},
            'steps':[
                {'range':[-0.5,0.5],'color':'lightgray'},
                {'range':[0.5,1.5],'color':'gray'},
                {'range':[1.5,2.5],'color':'lightblue'}],
            'threshold':{'line':{'color':'red','width':4}, 'thickness':0.75,'value':2.5}
        },
    ))

    return fig4

# Run app
if __name__=='__main__':
    app.run_server(debug=True, port=8045)