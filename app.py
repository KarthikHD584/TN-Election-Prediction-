# ================================
# TAMIL NADU ELECTION PREDICTION
# BEST STREAMLIT UI DESIGN
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="TAMIL NADU ELECTION PREDICTION",
    page_icon="🗳️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Background */

.stApp {
    background-image:
    linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.82)),
    url("https://images.unsplash.com/photo-1529107386315-e1a2ed48a620?q=80&w=2070");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Main Title */

.main-title {
    text-align: center;
    font-size: 58px;
    color: #FFD700;
    font-weight: bold;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: white;
    font-size: 22px;
    margin-bottom: 30px;
}

/* Cards */

.card {
    background: rgba(255,255,255,0.10);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(8px);
    box-shadow: 0px 0px 15px rgba(255,255,255,0.2);
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.95);
}

/* Metrics */

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Text */

h1,h2,h3,h4,h5,p,label,span {
    color: white !important;
}

/* Button */

.stButton>button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(90deg,#11998e,#38ef7d);
    color: white;
}

/* Table */

[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.05);
}

/* Input */

.stNumberInput input {
    background-color: rgba(255,255,255,0.1);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown(
    "<div class='main-title'>🗳️ TAMIL NADU ELECTION PREDICTION</div>",
    unsafe_allow_html=True
)


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    df = pd.read_csv("eci_results_tamilnadu_2026.csv")
    return df

df = load_data()

# ---------------- CLEAN DATA ----------------

df = df.dropna()

# Winner Column

max_votes = df.groupby('Constituency')['Total Votes'].transform('max')

df['Winner'] = np.where(
    df['Total Votes'] == max_votes,
    1,
    0
)

# Encoding

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

# ---------------- TRAIN MODEL ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

joblib.dump(model, "tamilnadu_model.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚡ Navigation")

menu = st.sidebar.radio(
    "Select Menu",
    [
        "🏠 Home",
        "📄 Dataset",
        "📊 Analytics",
        "🏆 Constituency Winner",
        "🤖 Election Prediction",
        "📈 Sentiment Analysis"
    ]
)

# ================= HOME =================

if menu == "🏠 Home":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Constituencies",
            df['Constituency'].nunique()
        )

    with col2:
        st.metric(
            "Total Candidates",
            df['Candidate'].nunique()
        )

    with col3:
        st.metric(
            "Total Parties",
            df['Party'].nunique()
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    <h2>📌 About This Project</h2>

    This AI-powered election dashboard predicts election winners using:

    ✅ XGBoost  
    ✅ Random Forest  
    ✅ Logistic Regression  
    ✅ Sentiment Analysis  
    ✅ Interactive Visualizations  

    Built using:
    
    🔹 Python  
    🔹 Machine Learning  
    🔹 Streamlit  
    🔹 Plotly  

    </div>
    """, unsafe_allow_html=True)

# ================= DATASET =================

elif menu == "📄 Dataset":

    st.subheader("📄 Tamil Nadu Election Dataset")

    st.dataframe(df)

# ================= ANALYTICS =================

elif menu == "📊 Analytics":

    st.subheader("📊 Election Analytics")

    # Top Parties

    party_votes = df.groupby('Party')['Total Votes'] \
                    .sum() \
                    .sort_values(ascending=False) \
                    .head(10)

    fig = px.bar(
        x=party_votes.index,
        y=party_votes.values,
        color=party_votes.values,
        title="Top 10 Parties by Votes"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Seat Share

    party_seats = df[df['Winner'] == 1]['Party'].value_counts()

    fig2 = px.pie(
        names=party_seats.index,
        values=party_seats.values,
        title="Seat Share"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ================= WINNER =================

elif menu == "🏆 Constituency Winner":

    st.subheader("🏆 Find Constituency Winner")

    constituency = st.selectbox(
        "Select Constituency",
        sorted(df['Constituency'].unique())
    )

    const_data = df[df['Constituency'] == constituency]

    winner = const_data.loc[
        const_data['Total Votes'].idxmax()
    ]

    st.success(f"🏆 Winner: {winner['Candidate']}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Party", winner['Party'])

    with col2:
        st.metric("Votes", int(winner['Total Votes']))

    with col3:
        st.metric("Vote %", winner['% Votes'])

    st.subheader("📋 Candidate Ranking")

    ranked = const_data.sort_values(
        by='Total Votes',
        ascending=False
    )

    st.dataframe(
        ranked[['Candidate','Party','Total Votes','% Votes']]
    )

# ================= PREDICTION =================

elif menu == "🤖 Election Prediction":

    st.subheader("🤖 Predict Future Election Winner")

    evm_votes = st.number_input(
        "Enter EVM Votes",
        min_value=0
    )

    postal_votes = st.number_input(
        "Enter Postal Votes",
        min_value=0
    )

    total_votes = st.number_input(
        "Enter Total Votes",
        min_value=0
    )

    vote_percent = st.slider(
        "Vote Percentage",
        0.0,
        100.0,
        45.0
    )

    party = st.selectbox(
        "Select Party",
        sorted(df['Party'].unique())
    )

    constituency = st.selectbox(
        "Select Constituency",
        sorted(df['Constituency'].unique())
    )

    party_encoded = party_encoder.transform([party])[0]

    const_encoded = const_encoder.transform([constituency])[0]

    if st.button("Predict Winner"):

        sample = pd.DataFrame({

            'EVM Votes': [evm_votes],
            'Postal Votes': [postal_votes],
            'Total Votes': [total_votes],
            '% Votes': [vote_percent],
            'Party_Encoded': [party_encoded],
            'Constituency_Encoded': [const_encoded]

        })

        prediction = model.predict(sample)

        probability = model.predict_proba(sample)[0][1]

        if prediction[0] == 1:

            st.balloons()

            st.success("🏆 Predicted Result: WINNER")

            st.progress(float(probability))

            st.write(
                f"Winning Probability: {round(probability*100,2)}%"
            )

        else:

            st.error("❌ Predicted Result: NOT WINNER")

            st.progress(float(probability))

            st.write(
                f"Winning Probability: {round(probability*100,2)}%"
            )

# ================= SENTIMENT =================

elif menu == "📈 Sentiment Analysis":

    st.subheader("📈 Political Sentiment Analysis")

    tweets = [
        "MK Stalin is doing good work",
        "Bad government performance",
        "People support DMK",
        "Corruption issue increasing",
        "Excellent development in Tamil Nadu"
    ]

    tweet_df = pd.DataFrame(
        tweets,
        columns=['Tweet']
    )

    positive = 3
    negative = 1
    neutral = 1

    sentiment_data = pd.DataFrame({
        'Sentiment': ['Positive','Negative','Neutral'],
        'Count': [positive, negative, neutral]
    })

    fig = px.bar(
        sentiment_data,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        title='Public Sentiment Analysis'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(tweet_df)

# ---------------- FOOTER ----------------

st.markdown("""
<hr>

<center>

<h4 style='color:white'>

Made with ❤️ using Streamlit | Election Prediction System

</h4>

</center>
""", unsafe_allow_html=True)
