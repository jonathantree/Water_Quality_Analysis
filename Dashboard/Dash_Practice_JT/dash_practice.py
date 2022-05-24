from dash import Dash, dcc, Output, Input
import dash_bootstrap_components as dbc

#Build Components
app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
my_text = dcc.Markdown(children='')
my_input = dbc.Input(value='# Hello World')

#Customize the layout
app.layout = dbc.Container([my_text, my_input])

#Add a callback
@app.callback(
    Output(my_text, component_property='children'),
    Input(my_input, component_property='value')
)

#Define a function to use the callback
def update_title(user_input):
    return user_input

#Run App
if __name__ == '__main__':
    app.run_server(port=8051)
