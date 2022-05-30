from dash import Dash, dcc, Output, Input, html  
import dash_bootstrap_components as dbc    
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd                       
from urllib.request import urlopen
import json
import pathlib

# app = Dash(__name__,
#                 external_stylesheets=[dbc.themes.FLATLY],
#                 meta_tags=[{'name': 'viewport',
#                             'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
#                 )
# Add a title
title=("Water Quality Analysis")    

#read in data
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
df = pd.read_csv(DATA_PATH.joinpath('new_priority_by_fips.csv'))

#add graphs

fig1 = px.scatter(
    df, 
    x="Simpson_Ethnic_DI", 
    y="Sum_ContaminantFactor", 
    color="Label",
    size='Num_Contaminants', 
    hover_data=['Sum_ContaminantFactor'],
    labels={
        "Sum_ContaminantFactor": "Total Conataminant Factor",
        "Simpson_Ethnic_DI" : " Simpson Ethnic Index"
                     
            },
)

fig1.update_layout(
    title={
        'text': "Plot of the Adaboost Top Feature Importance",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig2 = px.scatter(
    df, 
    x="Shannon_Race_DI", 
    y="Sum_ContaminantFactor", 
    color="Label",
    size='Num_Contaminants', 
    hover_data=['Sum_ContaminantFactor'],
    labels={
        "Sum_ContaminantFactor": "Total Conataminant Factor",
        "Shannon_Race_DI" : " Shannon Race Index"
                     
            },
)

fig2.update_layout(
    title={
        'text': "Plot of the Adaboost Top Feature Importance",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

# layout

layout = dbc.Container([
    
    dbc.Row( #Title
        dbc.Col([
            html.H1("Water Quality Analysis",
                    className='text-center text-primary mb-4'), #mb-4 padding
            html.H6("This page will talk about Machine Learning Results",
                    className="text-center text-muted")
        ], width=12)
    ),
    
    dbc.Row([
        dbc.Col([
            html.P("Row 1 column 1"),
            dbc.Card(
                dbc.CardBody(
                    html.P("card")
                )
            ),

        ], width=6),
        dbc.Col([
            html.P("Row 1 column 2"),
            dbc.Card(
                dbc.CardBody(
                    html.P("card")
                )
            ),

        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            # html.P("Row 2 column 1"),
            dbc.Card([
                dbc.CardHeader("Simpson Ethnic Diversity Index vs Total Contaminant Factor",className='card-header'),
                dbc.CardBody(
                    # html.P("card"),
                    dcc.Graph('scatter-1',figure=fig1)
                )
            ]),

        ], width=6),

        dbc.Col([
            # html.P("Row 2 column 2"),
            dbc.Card([
                dbc.CardHeader("Shannon Race Diversity Index vs. Total Contaminant Factor",className='card-header'),
                dbc.CardBody(
                    # html.P("card"),
                    dcc.Graph('scatter-2',figure=fig2)
                )
            ]),

        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.P("Row 3 column 1"),
            dbc.Card(
                dbc.CardBody(
                    html.P("card")
                )
            ),

        ], width=6),
        dbc.Col([
            html.P("Row 3 column 1"),
            dbc.Card(
                dbc.CardBody(
                    html.P("card")
                )
            ),

        ], width=6),
    ])

])

# @app.callback(
#     Output('scatter-1','figure'),
#     Output('scatter-2','figure'),
#     Input()
# )

# # Run app
# if __name__=='__main__':
#     app.run_server(debug=True, port=8046)