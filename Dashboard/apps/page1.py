from dash import Dash, dcc, Output, Input, html  
import dash_bootstrap_components as dbc    
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd                       
from urllib.request import urlopen
import json

# app = Dash(__name__,
#                 external_stylesheets=[dbc.themes.FLATLY],
#                 meta_tags=[{'name': 'viewport',
#                             'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
#                 )
# Add a title
title=("Water Quality Analysis")    

# layout

layout = dbc.Container([
    
    dbc.Row( #Title
        dbc.Col([
            html.H1("Water Quality Analysis",
                    className='text-center text-primary mb-4'), #mb-4 padding
            html.H6("This page will talk about initial analysis of the data",
                    className="text-center text-muted")
        ], width=12)
    ),
    
    dbc.Row([
        dbc.Col([
            # html.P("Row 1 column 1"),
            dbc.Card([
                # dbc.CardHeader("The Data"),
                dbc.CardBody([
                    html.H2("The Data", className='card-title'),
                    html.Li("2020 Dicennial US Census Data provided demographic information at the the county level for every state (except Washington.) From these we calculated the Simpson and Shannon Racial and Ethnic Diversity Indices ",className='class-text'),
                    html.Li("Income diversity data is from US Census American Community Survey Table B19083 - Gini Index.", className='class-text'),
                    html.Li("Water contamination data was scraped from the Environmental Working Group's Tapwater Database.", className='class-text')
                ])
            ]),
        ], width=5),
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
            html.P("Row 2 column 1"),
            dbc.Card(
                dbc.CardBody(
                    html.P("card")
                )
            ),

        ], width=6),
        dbc.Col([
            html.P("Row 2 column 2"),
            dbc.Card(
                dbc.CardBody(
                    html.P("card")
                )
            ),

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

# # Run app
# if __name__=='__main__':
#     app.run_server(debug=True, port=8046)