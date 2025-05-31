import streamlit as st
import os
import pandas as pd
import numpy as np
import openai
from scipy.stats import ttest_ind


def get_openai_api_key(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key

# Load OpenAI key and set environment
api_key = get_openai_api_key('api_key.txt')
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the data
data = pd.read_csv('daylio_export_2025_05_31.csv')

# Mapping mood to numeric values
mapping = {'meh': 3, 'good': 4, 'rad': 5, 'bad': 2}
data['mood'] = data['mood'].map(mapping)

# Parse 'full_date' column as datetime if not already
if not np.issubdtype(data['full_date'].dtype, np.datetime64):
    data['full_date'] = pd.to_datetime(data['full_date'])

# Split the activities column and explode into long format
data['activities_split'] = data['activities'].str.split('|')
df_long = data.explode('activities_split')

# Convert activities to lowercase and strip whitespaces
df_long['activities_split'] = df_long['activities_split'].str.lower().str.strip()

# Prepare data for chatbot context
df_long['note'] = df_long['note'].fillna('')
df_long['combined'] = df_long.apply(
    lambda row: f"On {row['full_date'].date()} ({row['weekday']} at {row['time']}), mood was {row['mood']}. Activities: {row['activities_split'] or 'None'}. Note: {row['note'] or 'None'}",
    axis=1
)

# Initialize session state for chatbot
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = []

if 'combined_texts' not in st.session_state:
    st.session_state['combined_texts'] = []

# Load precomputed embeddings once
def load_precomputed_embeddings():
    if len(st.session_state['embeddings']) == 0:
        st.session_state['embeddings'] = np.load('embeddings.npy')
        st.session_state['combined_texts'] = df_long['combined'].tolist()

load_precomputed_embeddings()

# Embedding and similarity helper functions
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_relevant_entries(query, top_n=5):
    query_embedding = get_embedding(query)
    similarities = [cosine_similarity(query_embedding, e) for e in st.session_state['embeddings']]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [st.session_state['combined_texts'][i] for i in top_indices]

# Set app title
st.title("Journal Chatbot & Mood Dashboard")

# Create tabs for Chatbot and Dashboard
tab1, tab2 = st.tabs(["Chatbot", "Dashboard"])

with tab1:
    # Chatbot tab content
    st.header("Chat with your Journal")

    # Display previous chat messages
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.write(message['content'])

    user_input = st.chat_input("Type your question here...")

    if user_input:
        st.session_state['messages'].append({'role': 'user', 'content': user_input})

        relevant_entries = find_relevant_entries(user_input, top_n=5)

        notes_content = "You are a helpful assistant. Here is a list of relevant journal entries:\n" + \
                        "\n".join(relevant_entries) + "\nPlease use this information to answer any questions."

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": notes_content},
                {"role": "user", "content": user_input}
            ]
        )

        assistant_message = response.choices[0].message.content
        st.session_state['messages'].append({'role': 'assistant', 'content': assistant_message})

        with st.chat_message('assistant'):
            st.write(assistant_message)

with tab2:
    st.header("Mood Dashboard")

    # Create 'Year-Month' column for grouping
    data['year_month'] = data['full_date'].dt.to_period('M')

    # Group by Year-Month: calculate average mood and count entries
    mood_monthly = data.groupby('year_month').agg(
        average_mood=('mood', 'mean'),
        entry_count=('mood', 'size')
    ).reset_index()

    # Convert Period to Timestamp for plotting and display
    mood_monthly['year_month'] = mood_monthly['year_month'].dt.to_timestamp()

    # Plot average mood by month
    st.line_chart(
        mood_monthly.set_index('year_month')['average_mood'],
        use_container_width=True,
        height=300
    )

    # Show table with average mood and entry count
    st.write("### Average Mood and Entry Count by Month")
    st.dataframe(
        mood_monthly.rename(
            columns={
                'year_month': 'Month',
                'average_mood': 'Average Mood',
                'entry_count': 'Number of Entries'
            }
        )
    )

    # Bonus: mood counts by month
    st.write("### Mood Counts by Month")
    mood_counts = data.groupby(['year_month', 'mood']).size().unstack(fill_value=0)
    mood_counts.index = mood_counts.index.to_timestamp()
    st.dataframe(mood_counts)

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats

    # Calculate monthly average moods (assuming df_long is your exploded dataframe)
    mood_monthly = df_long.copy()
    mood_monthly['year_month'] = pd.to_datetime(mood_monthly['full_date']).dt.to_period('M')
    monthly_avg = mood_monthly.groupby('year_month')['mood'].mean().reset_index()
    monthly_avg['year_month'] = monthly_avg['year_month'].dt.to_timestamp()

    # Plot histogram of monthly averages
    fig, ax = plt.subplots(figsize=(8,5))
    counts, bins, patches = ax.hist(monthly_avg['mood'], bins=10, density=True, color='skyblue', alpha=0.7, edgecolor='black')

    # Overlay normal distribution curve with same mean and std
    mean = monthly_avg['mood'].mean()
    std = monthly_avg['mood'].std()
    xmin, xmax = bins[0], bins[-1]
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std)
    ax.plot(x, p, 'r--', linewidth=2, label='Normal Distribution')

    # Labels and title
    ax.set_title('Histogram of Monthly Average Mood')
    ax.set_xlabel('Average Mood')
    ax.set_ylabel('Density')
    ax.legend()

    st.pyplot(fig)


    st.write("---")
    st.header("Hypothesis Testing: Compare Average Mood Between Two Months")

    # Dropdown to select months to compare
    months = mood_monthly['year_month'].dt.strftime('%Y-%m').tolist()
    month1 = st.selectbox("Select first month:", months, index=0)
    month2 = st.selectbox("Select second month:", months, index=1)

    if st.button("Run t-test"):

        # Convert selection back to Timestamp for filtering
        month1_ts = pd.Timestamp(month1)
        month2_ts = pd.Timestamp(month2)

        # Filter moods for the selected months
        moods_month1 = data[data['full_date'].dt.to_period('M') == month1_ts.to_period('M')]['mood']
        moods_month2 = data[data['full_date'].dt.to_period('M') == month2_ts.to_period('M')]['mood']

        # Check that both samples have enough data
        if len(moods_month1) < 2 or len(moods_month2) < 2:
            st.warning("Not enough data points in one or both months to perform t-test.")
        else:
            # Perform independent two-sample t-test
            t_stat, p_val = ttest_ind(moods_month1, moods_month2, equal_var=False)  # Welch's t-test

            # st.write(f"**Comparing {month1} vs {month2}**")
            # st.write(f"t-statistic = {t_stat:.3f}")
            # st.write(f"p-value = {p_val:.4f}")

            # Calculate avg mood and count for each month
            avg1 = moods_month1.mean()
            count1 = len(moods_month1)
            avg2 = moods_month2.mean()
            count2 = len(moods_month2)

            st.markdown(f"""
            **Comparing:** {month1} vs {month2}  

            | Metric        | {month1}        | {month2}        |
            |---------------|-----------------|-----------------|
            | Average Mood  | {avg1:.3f}      | {avg2:.3f}      |
            | Entry Count   | {count1}        | {count2}        |
            | t-statistic   | \multicolumn{{2}}{{c}}{{{t_stat:.3f}}}          |
            | p-value       | \multicolumn{{2}}{{c}}{{{p_val:.4f}}}          |
            """)


            if p_val < 0.05:
                st.success("Result: Significant difference in average mood between the two months (reject null hypothesis).")
            else:
                st.info("Result: No significant difference in average mood between the two months (fail to reject null hypothesis).")
