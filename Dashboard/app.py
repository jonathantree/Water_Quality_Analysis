from dash import Dash
import dash_bootstrap_components as dbc

# meta_tags are required for the app layout to be mobile responsive
app = Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
                )

server = app.server

# app.title=('Water Quality Analysis')