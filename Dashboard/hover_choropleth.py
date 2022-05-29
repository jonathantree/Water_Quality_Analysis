from dash import Dash, dcc, Output, Input, html  
import dash_bootstrap_components as dbc    
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd                       
import sqlite3
from urllib.request import urlopen
import json
import os
import dash_daq as daq

#import and clean data
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# Read data
df = pd.read_csv('census_contaminant_priority_by_zip.csv', dtype={"zip":str,"fips":str})
# file_to_read = os.path.join("..","..","Census_Data_Cleaning","zips_to_counties.csv")
zips_to_counties = pd.read_csv("../Census_Data_Cleaning/zips_to_counties.csv",dtype={"zip": str})
zips_to_counties["county_fips"] = zips_to_counties["county_fips"].astype(str).apply('{:0>5}'.format)
db = r'/Users/jennadodge/uofo-virt-data-pt-12-2021-u-b/Water_Quality_Analysis/Database/database.sqlite3'
conn = sqlite3.connect(db)
cursor = conn.cursor() # Create cursor object
contaminants_df = pd.read_sql_query("SELECT * FROM all_contaminants",conn)
conn.close()
contaminants_df["Zip"] = contaminants_df["Zip"].astype(str).str[:-2].apply('{:0>5}'.format) 
cont_fips_df = pd.merge(contaminants_df,zips_to_counties,how="left",left_on="Zip",right_on='zip')
print(contaminants_df.head())
df_map = df[['Simpson Race Diversity Index','Simpson Ethnic Diversity Index', 'Shannon Race Diversity Index',
       'Shannon Ethnic Diversity Index', 'Gini Index',
       'Number of Contaminants', 'Population Served',
       'Total Contaminant Factor']]

app = Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
                )
# Add a title
app.title=("Water Quality Analysis")                   
#--------------------------------------------------------------------------
# Customize the Layout

app.layout = dbc.Container([

    dbc.Row( #Title
        dbc.Col(html.H1("Water Quality Analysis",
                    className='text-center text-primary mb-4'), #mb-4 padding
            width=12)
    ),

    dbc.Row([

        dbc.Col([
            html.H5("Please Select A Value", className='mb-3'),
            dcc.Dropdown(id = 'dropdown',options=df_map.columns.values,
                        value='Gini Index',
                        clearable=False),
            dcc.Graph(id='mygraph', figure={}, clickData=None, hoverData=None, # By defualt, these are None, unless you specify otherwise.
                  config={
                      'staticPlot': False,     # True, False
                      'scrollZoom': False,      # True, False
                      'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                      'showTips': False,       # True, False
                      'displayModeBar': 'hover',  # True, False, 'hover'
                      'watermark': True,
                      # 'modeBarButtonsToRemove': ['pan2d','select2d'],
                        },), 
        ], width=12),
 
    ], justify='center'),


    dbc.Row([
        dbc.Col([
            # html.P("Please Select a State"), 
            # dcc.Dropdown(id='states_dropdown',options=[{'label': s, 'value': s} for s in sorted(contaminants_df.State.unique())],
            #             value='VT',
            #             clearable=False),
            dcc.Graph(id='myhist', figure={})
        ], xs=12, sm=12, md=12, lg=8, xl=8), # responsive column sizing

        dbc.Col([
            html.P("Zip Code:"),
            dcc.Input(id='zip_code',
                    type='number',
                    # placeholder='',
                    value=97124,
                    debounce = True  # initial value displayed when page first loads
                    ),
            dcc.Graph(id='gauge',figure={})
        ], xs=12, sm=12, md=12, lg=4, xl=4), # responsive column sizing
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
    
    return fig 

@app.callback(
    Output('myhist','figure'),
    Output('scatter','figure'),
    Input('mygraph','clickData'),
    # Input('states_dropdown','value')
)
def update_hist(click_data):
    # print(state_input)
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

        return fig2, fig3
    
    else:
        # print(f'click data: {click_data}')
        dff = cont_fips_df.copy()
        click_fips = click_data['points'][0]['location']
        dff = dff[dff['county_fips']==click_fips]
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

        return fig2, fig3


# @app.callback(
#     Output('scatter','figure'),
#     Input('states_dropdown','value')
# )

# def update_scatter(state_input):

#     c_df = contaminants_df.copy()
#     c_df2 = c_df[c_df.State == state_input]
#     top_c_df =c_df2.groupby(by=["Contaminant"]).sum().sort_values(by=['Contaminant_Factor'], ascending=False)[['People_served', 'Contaminant_Factor']]
#     top15_c_df = top_c_df.head(15)
#     top15_c_df = top15_c_df.reset_index()

#     fig3 = px.scatter(
#         data_frame = top15_c_df, 
#         x="Contaminant", 
#         y="People_served",
#         size="Contaminant_Factor", 
#         color="Contaminant",
#         hover_name="Contaminant", 
#         size_max=60
#     )
#     return fig3

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
            'axis':{'range':[-0.25,1.25]},
            'bar':{'color':'teal'},
            'steps':[
                {'range':[0,0.5],'color':'lightgray'},
                {'range':[0.5,1.0],'color':'gray'}]
        },))

    return fig4

# Run app
if __name__=='__main__':
    app.run_server(debug=True, port=8045)