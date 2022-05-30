from dash import Dash, dcc, Output, Input, html  
import dash_bootstrap_components as dbc    
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd                       
from urllib.request import urlopen
import json

app = Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
                )
# Add a title
app.title=("Water Quality Analysis")    

# layout

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col()
    ])

])

# Run app
if __name__=='__main__':
    app.run_server(debug=True, port=8046)