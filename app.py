import pandas as pd
import streamlit as st
import plotly.express as px

# Load the data
data = pd.read_csv('daylio_export_2024_08_06.csv')

# Mapping mood to numeric values
mapping = {'meh': 3, 'good': 4, 'rad': 5, 'bad': 2}
data['mood'] = data['mood'].map(mapping)

# Split the activities column and explode into long format
data['activities_split'] = data['activities'].str.split('|')
df_long = data.explode('activities_split')

# Convert activities to lowercase and strip whitespaces
df_long['activities_split'] = df_long['activities_split'].str.lower().str.strip()

# Group by activities and calculate the mean mood
result = df_long.groupby('activities_split').agg(
    mean_mood=('mood', 'mean'),
    activity_count=('mood', 'count')
).sort_values('activity_count', ascending=False).reset_index()

# Create a bar plot
fig = px.bar(result, x='activities_split', y='activity_count', title='Activity Count')

# Streamlit app layout
st.title('Activity Dashboard')
st.plotly_chart(fig)