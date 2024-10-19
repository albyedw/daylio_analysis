import pandas as pd
import streamlit as st
import plotly.express as px
import openai
import numpy as np
import os

def get_openai_api_key(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key

# Example usage
api_key = get_openai_api_key('api_key.txt')

# Set up the OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = os.getenv("OPENAI_API_KEY")



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


st.title("Journal Chatbot")

# Initialize session state to store chat history and embeddings
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = []
    
if 'combined_texts' not in st.session_state:
    st.session_state['combined_texts'] = []

# Load the full dataset
df_full = df_long

df_full['note'] = df_full['note'].fillna('')
# Combine relevant columns into a single string for each row
df_full['combined'] = df_full.apply(lambda row: f"On {row['full_date']} ({row['weekday']} at {row['time']}), mood was {row['mood']}. Activities: {row['activities_split'] or 'None'}. Note: {row['note'] or 'None'}", axis=1)

# Function to load precomputed embeddings from a file
def load_precomputed_embeddings():
    if len(st.session_state['embeddings']) == 0:
        # Load the embeddings from the .npy file
        st.session_state['embeddings'] = np.load('embeddings.npy')
        # Load the combined texts from a CSV or directly from df_full if you don't have a saved file
        st.session_state['combined_texts'] = df_full['combined'].tolist()
        print("Embeddings and texts loaded successfully.")

# Function to get embedding for a single text
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to find the most relevant journal entries based on cosine similarity
def find_relevant_entries(query, top_n=5):
    query_embedding = get_embedding(query, model='text-embedding-ada-002')
    similarities = [cosine_similarity(query_embedding, entry_embedding) for entry_embedding in st.session_state['embeddings']]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [st.session_state['combined_texts'][i] for i in top_indices]

# Load precomputed embeddings when the app starts (this will run once)
load_precomputed_embeddings()

# Display chat messages from history
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to chat history
    st.session_state['messages'].append({'role': 'user', 'content': user_input})

    # Retrieve the top 5 most relevant journal entries based on the query
    relevant_entries = find_relevant_entries(user_input, top_n=5)

    # Combine the relevant entries as context
    notes_content = f"You are a helpful assistant. Here is a list of relevant journal entries: \n" + "\n".join(relevant_entries) + "\n Please use this information to answer any questions."

    # Generate response using OpenAI API
    response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": notes_content},
        {"role": "user", "content": user_input}
    ]
)


    # Add assistant message to chat history
    assistant_message = response.choices[0].message.content
    st.session_state['messages'].append({'role': 'assistant', 'content': assistant_message})

    # Display assistant message
    with st.chat_message('assistant'):
        st.write(assistant_message)
