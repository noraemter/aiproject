import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained Random Forest model
model = load_model('traffic_congestion_model.keras')

# Load the dataset
data = pd.read_csv('dataset/urban_mobility_data_past_year.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

min_temp = data['temperature'].min()
max_temp = data['temperature'].max()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Traffic Congestion Prediction Dashboard", className="text-center")),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Select Date:"),
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=data['timestamp'].min().date(),
                max_date_allowed=data['timestamp'].max().date(),
                date=data['timestamp'].min().date()  
            ),
        ], width=6),
        dbc.Col([
            html.Label("Select Time:"),
            dcc.Slider(
                id='time-slider',
                min=0,
                max=23,
                step=1,
                value=12,  # Default to noon
                marks={i: f'{i}:00' for i in range(0, 24)}
            ),
        ], width=6),
    ], className="mt-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Weather Condition:"),
            dcc.Dropdown(
                id='weather-dropdown',
                options=[
                    {'label': 'Clear', 'value': 'Clear'},
                    {'label': 'Cloudy', 'value': 'Cloudy'},
                    {'label': 'Rainy', 'value': 'Rainy'}
                ],
                value='Clear'
            ),
        ], width=6),
 dbc.Col([
            html.Label("Select Temperature (°C):"),
            dcc.Slider(
                id='temperature-slider',
                min=min_temp,
                max=max_temp,
                step=1,
                value=(min_temp + max_temp) / 2, 
                marks={i: f'{i}°C' for i in range(int(min_temp), int(max_temp) + 1, 5)}
            ),
        ], width=6),
    ], className="mt-4"),
    dbc.Row([
        dbc.Col([
            dbc.Button("Predict Congestion", id="predict-btn", className="mt-4"),
            html.Div(id='prediction-output')
        ], width=12),
    ], className="mt-4"),
], fluid=True)

@app.callback(
    Output('prediction-output', 'children'),
    [Input('date-picker', 'date'),
     Input('time-slider', 'value'),
     Input('weather-dropdown', 'value'),
     Input('temperature-slider', 'value'),
     Input('predict-btn', 'n_clicks')]
)
def predict_congestion(selected_date, selected_time, weather_condition, temperature, n_clicks):
    if n_clicks is not None:
        try:

            weather_map = {
                'Clear': [1, 0, 0],
                'Cloudy': [0, 1, 0],
                'Rainy': [0, 0, 1]
            }
            weather_encoded = weather_map.get(weather_condition, [0, 0, 0])

            features = np.array(weather_encoded + [temperature]).reshape(1, -1)

            predicted_congestion = model.predict(features)[0][0] 

            return f'Predicted Congestion Level on {selected_date} at {selected_time}:00: {predicted_congestion:.2f}'
        except Exception as e:
            return f'Error: {str(e)}'
    return ''


if __name__ == '__main__':
    app.run_server(debug=True)
