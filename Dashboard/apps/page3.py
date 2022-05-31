from dash import Dash, dcc, Output, Input, html  
import dash_bootstrap_components as dbc
from matplotlib import container    
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd                       
# import sqlite3
from urllib.request import urlopen
import json
# import dash_daq as daq
import pathlib
from app import app

#import and clean data
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# Read data
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
df = pd.read_csv(DATA_PATH.joinpath('census_contaminant_priority_by_zip.csv'), dtype={"zip":str,"fips":str})

cont_fips_df = pd.read_csv(DATA_PATH.joinpath('all_contaminants_with_fips.csv'),dtype = {"zip":str,"county_fips":str})


df_map = df[['Simpson Race Diversity Index','Simpson Ethnic Diversity Index', 'Shannon Race Diversity Index',
       'Shannon Ethnic Diversity Index', 'Gini Index',
       'Number of Contaminants', 'Population Served',
       'Total Contaminant Factor']]


title=("Water Quality Analysis")                   
#--------------------------------------------------------------------------
# Customize the Layout

layout = dbc.Container([

    dbc.Row([ #Title
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Please select a value from the dropdown to view demographic "
                        "or contaminant data on a County level. Click on a County to view "
                        "more detailed graphs below.",
                        className="text-center card-subtitle")
                    ],
                className="card bg-light")
            ]),
            # html.H5("Initial Analysis",
                    # className='text-center text-primary mb-4'), #mb-4 padding
            # html.H6("This page will talk about initial analysis of the data",
            #         className="text-center text-muted")
        ], width=6),
        
        # dbc.Col(
        #     html.H5("Please select a value from the dropdown to view demographic "
        #                 "or contaminant data on a County level. Click on a County to view "
        #                 "more detailed graphs below.",
        #             className='text-center text-primary mb-4'), #mb-4 padding
        #     width=6),
        dbc.Col(
             dcc.Dropdown(id = 'dropdown',options=df_map.columns.values,
                        value='Number of Contaminants',
                        clearable=False),
            width = 6)
    ]),

    dbc.Row([

        dbc.Col([
            # html.H5("Please Select A Value", className='mb-3'),
            # dcc.Dropdown(id = 'dropdown',options=df_map.columns.values,
            #             value='Number of Contaminants',
            #             clearable=False),
            dcc.Graph(id='mygraph', figure={}, clickData=None, hoverData=None, # By defualt, these are None, unless you specify otherwise.
                  config={
                      'staticPlot': False,     # True, False
                      'scrollZoom': True,      # True, False
                      'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                      'showTips': False,       # True, False
                      'displayModeBar': 'hover',  # True, False, 'hover'
                      'watermark': True,
                      # 'modeBarButtonsToRemove': ['pan2d','select2d'],
                        },), 
        ], className='mb-4', width=12),
 
    ], className='mb-5', justify='center'),

    dbc.Row([
        dbc.Col([
            # dcc.Markdown(id='county_confirm', children=[])#, className='text-center text-primary mb-4'),
            html.H4("The following charts are based on the County chosen above.",className='text-center text-primary mb-3')
        ])
    ], className='mb-5', justify='center'),

    dbc.Row([
        dbc.Col([
            # html.H4("Top 10 Contaminants ",className='text-center text-primary mb-3'),
            dcc.Graph(id='myhist', figure={})
        ], xs=12, sm=12, md=12, lg=8, xl=8), # responsive column sizing

        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H4("Priority Level", className='card-title text-center'),
                    html.P('Based on the demographic and water quality data', className='card-text text-center'),
                    dcc.Graph(id='gauge',figure={}),

                ]), color="info"
            )
        ], xs=12, sm=12, md=12, lg=4, xl=4), # responsive column sizing
    ], className='mb-5', justify='center'),
    
    dbc.Row([
        dbc.Col([
            html.H4('Top Contaminants vs People Served',className='text-center text-primary mb-3'), 
            html.H6('Bubble size indicates Contaminant Factor',className="text-center text-muted mb-2")],
        width=12),
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
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig 

@app.callback(
    Output('myhist','figure'),
    Output('scatter','figure'),
    Output('gauge','figure'),
    # Output('county_confirm','value'),
    Input('mygraph','clickData'),
)
def update_hist(click_data):
    print(f'click data: {click_data}')
    if click_data is None:
        dff = cont_fips_df.copy()
        dff = dff[dff['county_fips']=='41005']
        dff2 = dff.groupby(["Contaminant"]).size().to_frame().sort_values([0], ascending = True).tail(10).reset_index()
        dff2 = dff2.rename(columns={0: 'Count of Contaminant'})

        fig2 = px.histogram(
            data_frame = dff2, 
            x = 'Count of Contaminant', 
            y="Contaminant").update_layout(
            title={"text": "Top Ten Contaminants", "x": 0.5}, 
            xaxis_title="Number of Occurences"
        )

        dff3 = dff.groupby(by=["Contaminant"]).sum().sort_values(by=['Contaminant_Factor'], ascending=False)[['People_served', 'Contaminant_Factor']]
        top15_c_df = dff3.head(15)
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

        df_copy = df.copy()
        df_copy = df_copy[df_copy['fips']=='41005']

        fig4 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value=df_copy.Priority.values[0],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text':'Priority','font':{'size':24}},
            gauge = {
                'axis':{'range':[-0.25,1.25]},
                'bar':{'color':'teal'},
                'steps':[
                    {'range':[0,0.5],'color':'lightgray'},
                    {'range':[0.5,1.0],'color':'gray'}]
            },))

        #container=f'The Below Graphs are Based on Clackamas County'

        return fig2, fig3, fig4#, container
    
    else:
        #print(f'click data: {click_data}')
        dff = cont_fips_df.copy()
        click_fips = click_data['points'][0]['location']
        click_county = click_data['points'][0]['customdata'][0]
        dff = dff[dff['county_fips']==click_fips]
        dff2 = dff.groupby(["Contaminant"]).size().to_frame().sort_values([0], ascending = True).tail(10).reset_index()
        dff2 = dff2.rename(columns={0: 'Count of Contaminant'})
        

        fig2 = px.histogram(
            data_frame = dff2, 
            x = 'Count of Contaminant', 
            y="Contaminant").update_layout(
            # title={"text": f"Top Ten Contaminants in {click_county}", "x": 0.5}, 
            title=f'Top Ten Contaminants in {click_county}',
            xaxis_title="Number of Occurences"
        )

        dff3 = dff.groupby(by=["Contaminant"]).sum().sort_values(by=['Contaminant_Factor'], ascending=False)[['People_served', 'Contaminant_Factor']]
        top15_c_df = dff3.head(15)
        top15_c_df = top15_c_df.reset_index()

        fig3 = px.scatter(
            data_frame = top15_c_df, 
            x="Contaminant", 
            y="People_served",
            size="Contaminant_Factor", 
            color="Contaminant",
            hover_name="Contaminant", 
            size_max=60,
            # title={'text':f'Contaminants in {click_county}',"size":24}
            )

        df_copy = df.copy()
        df_copy = df_copy[df_copy['fips']==click_fips]

        fig4 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value=df_copy.Priority.values[0],
            domain={'x': [0, 1], 'y': [0, 1]},
            # title={'text':f'Priority Level for {click_county}','font':{'size':24}},
            gauge = {
                'axis':{'range':[-0.25,1.25]},
                'bar':{'color':'teal'},
                'steps':[
                    {'range':[0,0.5],'color':'lightgray'},
                    {'range':[0.5,1.0],'color':'gray'}]
            },))
        
        #container=f"The Below Graphs Are Based on {click_county}"

        return fig2, fig3, fig4#, container