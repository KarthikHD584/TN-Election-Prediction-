# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Tamil Nadu Election Prediction",
    page_icon="🗳️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #fff5f5, #f0f8ff);
}

.stApp {
    background-image: url('https://images.unsplash.com/photo-1529101091764-c3526daf38fe');
    background-size: cover;
    background-attachment: fixed;
}

.title-style {
    font-size:50px;
    font-weight:bold;
    color:white;
    text-align:center;
    background: rgba(0,0,0,0.6);
    padding:20px;
    border-radius:15px;
}

.card {
    background-color: rgba(255,255,255,0.9);
    padding:20px;
    border-radius:15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}

.metric-box {
    background: linear-gradient(to right, #ff512f, #dd2476);
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
    font-size:22px;
    font-weight:bold;
}

.sidebar .sidebar-content {
    background: #111827;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown("""
<div class="title-style">
Tamil Nadu Election Prediction System
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- LOAD DATA ----------------

df = pd.read_csv("eci_results_tamilnadu_2026.csv")

df = df.dropna()

# ---------------- CREATE WINNER COLUMN ----------------

max_votes = df.groupby('Constituency')['Total Votes'].transform('max')

df['Winner'] = np.where(df['Total Votes'] == max_votes, 1, 0)

# ---------------- ENCODING ----------------

party_encoder = LabelEncoder()
df['Party_Encoded'] = party_encoder.fit_transform(df['Party'])

const_encoder = LabelEncoder()
df['Constituency_Encoded'] = const_encoder.fit_transform(df['Constituency'])

# ---------------- FEATURES ----------------

X = df[[
    'EVM Votes',
    'Postal Votes',
    'Total Votes',
    '% Votes',
    'Party_Encoded',
    'Constituency_Encoded'
]]

y = df['Winner']

# ---------------- SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL ----------------

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

# ---------------- SIDEBAR ----------------

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/6/6b/Election_icon.png",
    width=150
)

st.sidebar.title("Election Dashboard")

menu = st.sidebar.radio(
    "Select Option",
    [
        "Home",
        "Prediction",
        "Party Analysis",
        "Constituency Winner",
        "Sentiment Analysis"
    ]
)

# ---------------- HOME ----------------

if menu == "Home":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
        Total Candidates<br>
        {len(df)}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
        Political Parties<br>
        {df['Party'].nunique()}
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
        Model Accuracy<br>
        {round(accuracy*100,2)}%
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    party_votes = df.groupby('Party')['Total Votes'].sum().sort_values(ascending=False).head(10)

    fig = px.bar(
        x=party_votes.index,
        y=party_votes.values,
        title="Top 10 Parties by Votes",
        labels={'x':'Party', 'y':'Votes'}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------

elif menu == "Prediction":

    st.subheader("Election Winning Prediction")

    evm_votes = st.number_input("EVM Votes", 0, 500000, 70000)

    postal_votes = st.number_input("Postal Votes", 0, 10000, 500)

    total_votes = st.number_input("Total Votes", 0, 500000, 70500)

    percent_votes = st.slider("Vote Percentage", 0.0, 100.0, 45.0)

    party = st.selectbox(
        "Select Party",
        df['Party'].unique()
    )

    constituency = st.selectbox(
        "Select Constituency",
        df['Constituency'].unique()
    )

    if st.button("Predict Result"):

        party_encoded = party_encoder.transform([party])[0]

        const_encoded = const_encoder.transform([constituency])[0]

        sample = pd.DataFrame({
            'EVM Votes': [evm_votes],
            'Postal Votes': [postal_votes],
            'Total Votes': [total_votes],
            '% Votes': [percent_votes],
            'Party_Encoded': [party_encoded],
            'Constituency_Encoded': [const_encoded]
        })

        result = model.predict(sample)

        probability = model.predict_proba(sample)[0][1]

        st.write("")

        if result[0] == 1:
            st.success(f"High Winning Chance — Probability: {round(probability*100,2)}%")
            st.balloons()

        else:
            st.error(f"Low Winning Chance — Probability: {round(probability*100,2)}%")

# ---------------- PARTY ANALYSIS ----------------

elif menu == "Party Analysis":

    st.subheader("Party Wise Seat Analysis")

    party_seats = df[df['Winner'] == 1]['Party'].value_counts()

    fig = px.pie(
        names=party_seats.index,
        values=party_seats.values,
        title="Seat Share Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("")

    fig2 = px.bar(
        x=party_seats.index,
        y=party_seats.values,
        color=party_seats.index,
        title="Party Wise Seats"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ---------------- CONSTITUENCY WINNER ----------------

elif menu == "Constituency Winner":

    st.subheader("Find Constituency Winner")

    constituency_name = st.selectbox(
        "Choose Constituency",
        sorted(df['Constituency'].unique())
    )

    const_data = df[df['Constituency'] == constituency_name]

    winner = const_data.loc[const_data['Total Votes'].idxmax()]

    st.markdown(f"""
    <div class="card">
    <h2>Winning Candidate: {winner['Candidate']}</h2>
    <h3>Party: {winner['Party']}</h3>
    <h3>Total Votes: {winner['Total Votes']}</h3>
    </div>
    """, unsafe_allow_html=True)

    ranked = const_data.sort_values(by='Total Votes', ascending=False)

    st.write("")

    st.dataframe(
        ranked[['Candidate', 'Party', 'Total Votes']],
        use_container_width=True
    )

# ---------------- SENTIMENT ANALYSIS ----------------

elif menu == "Sentiment Analysis":

    st.subheader("Election Sentiment Analysis")

    tweets = [
        "MK Stalin is doing good work",
        "Bad government performance",
        "People support DMK",
        "Corruption issue increasing",
        "Excellent development in Tamil Nadu"
    ]

    df_tweets = pd.DataFrame(tweets, columns=['Tweet'])

    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):

        score = analyzer.polarity_scores(text)

        if score['compound'] >= 0.05:
            return 'Positive'

        elif score['compound'] <= -0.05:
            return 'Negative'

        else:
            return 'Neutral'

    df_tweets['Sentiment'] = df_tweets['Tweet'].apply(get_sentiment)

    st.dataframe(df_tweets, use_container_width=True)

    sentiment_count = df_tweets['Sentiment'].value_counts()

    fig = px.bar(
        x=sentiment_count.index,
        y=sentiment_count.values,
        color=sentiment_count.index,
        title="Public Sentiment Analysis"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------

st.write("")
st.markdown("""
<center>
<h4>Election Prediction Analytics Dashboard</h4>
</center>
""", unsafe_allow_html=True)
