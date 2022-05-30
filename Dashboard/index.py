from dash import Dash, dcc, Output, Input, html  
import dash_bootstrap_components as dbc   

# Connect to main app.py file
from app import app     #import app object from app.py file 
from app import server    # impor server object from app.py

# Connect to your app pages
from apps import page1, page2, page3 #put other app page names here

app.layout = dbc.Container([
    dcc.Location(id='url',refresh=False),
    dbc.NavbarSimple(    
        children=[
            # dbc.NavbarBrand("Water Quality Analysis", href='#'),
            dbc.Nav(
                [
                dbc.NavLink('Exploratory Data Analysis', href='/apps/page1'),
                dbc.NavLink('Machine Learning Model',href='/apps/page2'),
                dbc.NavLink('Results',href='/apps/page3')
                ]
            ),
        ],
    fluid=True, color = "light"
    ),    
    html.Div(id='page-content', children=[]), #page content all goes in here
    # dbc.Row([
    #     dbc.Col([
    #         dcc.Link('Exploratory Data Analysis | ', href='/apps/page1'),
    #         dcc.Link('Machine Learning Model | ',href='/apps/page2'),
    #         dcc.Link('Results',href='/apps/page3'),
    #     ]),
    # ]),
])

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/apps/page1':
        return page1.layout
    if pathname == '/apps/page2':
        return page2.layout
    if pathname == '/apps/page3':
        return page3.layout
    else:
        return page1.layout


if __name__ == '__main__':
    app.run_server(debug=False, port=8049)