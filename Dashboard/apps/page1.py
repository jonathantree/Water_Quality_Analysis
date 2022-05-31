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
            dbc.Card([
                dbc.CardBody([
                    html.H3("Do communities with traditionally underserved demographics have access to clean drinking water? "
                        "If so, how can we prioritize which communities are most in the need of cleaner water?",
                        className="text-center card-subtitle mb-4")
                    ],
                className="card text-white bg-info mb-4")
            ]),
            # html.H5("Initial Analysis",
                    # className='text-center text-primary mb-4'), #mb-4 padding
            # html.H6("This page will talk about initial analysis of the data",
            #         className="text-center text-muted")
        ], width=12), justify="center"
    ),
    
    dbc.Row([
        dbc.Col([
            # html.P("Row 1 column 1"),
            dbc.Card([
                # dbc.CardHeader("The Data"),
                dbc.CardBody([
                    html.H3("The Data", className='card-title mb-4'),
                    html.Li("2020 Dicennial US Census Data provided demographic information at the the county level"
                    " for every state (except Washington). From these we calculated the Simpson and Shannon Racial and "
                    "Ethnic Diversity Indices.",className='class-text mb-2'),
                    html.Li("Income diversity data is from US Census American Community Survey Table B19083 - Gini Index.", className='class-text mb-2'),
                    html.Li("Using Selenium and Chrome Webdriver, we scraped water contamination data from the Environmental Working Group's Tapwater Database. "
                            "We collected over 120,000 records from utility companies 27 states.", className='class-text mb-2'),
                    html.Li("The Environmental Working Group provides its own recommendations for safe contaminant levels which "
                            "can be quite different from the Environmental Protection Agency's recommendations. In many cases the EPA "
                            "had no recommended level for safety.", className='class-text mb-2'),
                    html.Li("The Contaminant Factor (e.g. a contaminant has 100x the limit recommended by the EWG) is a measure of the severity "
                        "of an individual contaminant in a water supply. The Total Contaminant Factor is a sum of the individual "
                        "contaminant factors for a given Zip Code.", className='class-text mb-2')
                ])
            ]),
        ], xs=12, sm=12, md=12, lg=6, xl=6),
        dbc.Col([
            # html.P("Row 1 column 2"),
            dbc.Card([
                dbc.CardBody([
                    html.H3("Tests for Normality", className='card-title mb-4'),
                    html.Li("No features of the data pass the Shapiro-Wilk test for normality (Shapiro.test() in R)",className='class-text mb-2'),
                    html.Li("Many Features show a F-distribution, nulling any parametric tests for correlation, such as the Shapiro-Wilk test.", className='class-text mb-2'),
                    html.Li("Visualization of data to look for trends show that most data is right skewed.", className='class-text mb-2'),
                    html.Li("The Contaminant data (in particular the Sum Contaminant Factor, Average, and Number of Contaminants) "
                            "had significant outliers that we decided to eliminate.", className='class-text mb-2'),
                    dbc.CardImg(src="/assets/Num_Cont_distribution.png", top=False, bottom=True,
                    title="Distribution of Number of Contaminants", alt='Distribution of Number of Contaminants')
                ])
            ]),
        ], xs=12, sm=12, md=12, lg=6, xl=6)
    ], className='mb-5', justify="evenly"),

    dbc.Row([
        dbc.Col([
            html.H5("Significant outliers can be seen in exploratory data analysis. These outliers "
                    "make it a challenge to visualize the data.",className='text-center text-primary mb-2'),
            # dbc.Card(
            #     dbc.CardBody(
            #         html.P("card")
            #     )
            # ),

        ], xs=12, sm=12, md=12, lg=6, xl=6),
        dbc.Col([
            html.H5("After removing outliers, the data has a more normal distribution. "
                    "Less than 10% of data points were eliminated in this process.",className='text-center text-primary mb-2'),
            # dbc.Card(
            #     dbc.CardBody(
            #         html.P("card")
            #     )
            # ),

        ], xs=12, sm=12, md=12, lg=6, xl=6),
    ], className='mb-5', justify="evenly"),

    dbc.Row([
        dbc.Col([
            # html.H3("Row 2 column 1"),
            dbc.Card(
                dbc.CardImg(src="/assets/Sum_Contaminant_Factor_Boxplot_before.png", top=True, bottom=False,
                    title="Total Contaminant Factor Boxplot Before", alt='Total Contaminant Factor Boxplot Before')
            ),
        ], xs=12, sm=6, md=6, lg=3, xl=3),
        dbc.Col([
            # html.P("Row 2 column 2"),
            dbc.Card(
                dbc.CardImg(src="/assets/Sum_Contaminant_Factor_Hist_before.png", top=False, bottom=True,
                    title="Total Contaminant Factor Histogram Before", alt='Total Contaminant Factor Histogram Before')
            ),
        ], xs=12, sm=6, md=6, lg=3, xl=3),
        dbc.Col([
            # html.P("Row 2 column 1"),
            dbc.Card(
                dbc.CardImg(src="/assets/Sum_Contaminant_Factor_Boxplot_after.png", top=True, bottom=False,
                    title="Total Contaminant Factor Boxplot Before", alt='Total Contaminant Factor Boxplot Before')
            ),
        ], xs=12, sm=6, md=6, lg=3, xl=3),
        dbc.Col([
            # html.P("Row 2 column 2"),
            dbc.Card(
                dbc.CardImg(src="/assets/Sum_Contaminant_Factor_Hist_after.png", top=False, bottom=True,
                    title="Total Contaminant Factor Histogram Before", alt='Total Contaminant Factor Histogram Before')
            ),
        ], xs=12, sm=6, md=6, lg=3, xl=3),
    ], className='mb-5', justify="evenly"),

])

# # Run app
# if __name__=='__main__':
#     app.run_server(debug=True, port=8046)