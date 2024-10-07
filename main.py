import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open('pipe.pkl', 'rb') as f:
    pipe = pickle.load(f)

# Streamlit app
st.title('IPL Win Predictor')

# Input fields
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah',  'Mohali', 'Bengaluru']

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city', sorted(cities))
target = st.number_input('Target', min_value=0)
runs_left = st.number_input('Runs Left', min_value=0)
balls_left = st.number_input('Balls Left', min_value=0)
wickets = st.number_input('Wickets Left', min_value=0)
#crr = st.number_input('Current Run Rate', min_value=0.0)
#rrr = st.number_input('Required Run Rate', min_value=0.0)

# Prediction button
if st.button('Predict Probability'):
    # Create input data dictionary
    input_data = {
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [(target-runs_left)/(120-balls_left)],
        'rrr': [runs_left/balls_left]
    }

    # Create DataFrame
    input_df = pd.DataFrame(input_data)

    # Make prediction
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")