# =========================================================
# PREMIUM TAMIL NADU ELECTION PREDICTION
# ATTRACTIVE STREAMLIT UI WITH WINNING CELEBRATION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Tamil Nadu Election Prediction",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* MAIN APP BACKGROUND */

.stApp {
    background-image:
    linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.82)),
    url("https://images.unsplash.com/photo-1541872705-1f73c6400ec9?q=80&w=2070");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* REMOVE STREAMLIT HEADER */

header {
    visibility: hidden;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#000000,#1c1c1c);
    border-right: 2px solid rgba(255,255,255,0.1);
}

/* MAIN TITLE */

.main-title {
    text-align: center;
    font-size: 65px;
    font-weight: bold;
    color: #FFD700;
    text-shadow: 2px 2px 15px rgba(255,215,0,0.8);
    margin-top: -30px;
}

/* SUB TITLE */

.sub-title {
    text-align: center;
    color: white;
    font-size: 24px;
    margin-bottom: 35px;
}

/* GLASS CARDS */

.glass {
    background: rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0px 0px 25px rgba(255,255,255,0.15);
}

/* METRIC BOX */

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0px 0px 15px rgba(255,255,255,0.08);
}

/* TEXT COLORS */

h1,h2,h3,h4,h5,p,label,span {
    color: white !important;
}

/* BUTTON DESIGN */

.stButton>button {

    background: linear-gradient(
    90deg,
    #ff512f,
    #dd2476
    );

    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 14px;
    border: none;
    padding: 14px 30px;
    width: 100%;
    transition: 0.4s;
}

.stButton>button:hover {

    background: linear-gradient(
    90deg,
    #11998e,
    #38ef7d
    );

    transform: scale(1.03);
}

/* INPUTS */

.stTextInput input,
.stNumberInput input {
    background-color: rgba(255,255,255,0.1);
    color: white;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

/* WINNER TEXT */

.winner-box {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00ff88;
    text-shadow: 0px 0px 15px #00ff88;
}

/* LOSER TEXT */

.loser-box {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #ff4b4b;
    text-shadow: 0px 0px 15px #ff4b4b;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown("""
<div class='main-title'>
🗳️ Tamil Nadu Election Prediction
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sub-title'>
AI Powered Election Prediction • Live Analytics • Winning Probability
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("eci_results_tamilnadu_2026.csv")

df = load_data()

df = df.dropna()

# =========================================================
# CREATE WINNER COLUMN
# =========================================================

max_votes = df.groupby('Constituency')['Total Votes'].transform('max')

df['Winner'] = np.where(
    df['Total Votes'] == max_votes,
    1,
    0
)

# =========================================================
# ENCODING
# =========================================================

party_encoder = LabelEncoder()
df['Party_Encoded'] = party_encoder.fit_transform(df['Party'])

const_encoder = LabelEncoder()
df['Constituency_Encoded'] = const_encoder.fit_transform(df['Constituency'])

# =========================================================
# MODEL
# =========================================================

X = df[[
    'EVM Votes',
    'Postal Votes',
    'Total Votes',
    '% Votes',
    'Party_Encoded',
    'Constituency_Encoded'
]]

y = df['Winner']

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

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=120
)

st.sidebar.title("Election Navigation")

menu = st.sidebar.radio(
    "Choose Menu",
    [
        " Home",
        " Analytics",
        " Constituency Winner",
        " Election Prediction",
        " Vote Share"
    ]
)

# =========================================================
# HOME
# =========================================================

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
    <div class='glass'>

    <h2>📌 Project Overview</h2>

    This Election AI Dashboard predicts election winners using:
    
    ✅ XGBoost Machine Learning  
    ✅ Election Vote Analysis  
    ✅ Winning Probability Prediction  
    ✅ Interactive Graph Analytics  
    ✅ AI-based Election Intelligence  

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# ANALYTICS
# =========================================================

elif menu == "📊 Analytics":

    st.subheader("📊 Election Analytics Dashboard")

    party_votes = df.groupby('Party')['Total Votes'] \
                    .sum() \
                    .sort_values(ascending=False) \
                    .head(10)

    fig = px.bar(
        x=party_votes.index,
        y=party_votes.values,
        color=party_votes.values,
        text=party_votes.values,
        title="Top 10 Parties by Total Votes"
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# CONSTITUENCY WINNER
# =========================================================

elif menu == "🏆 Constituency Winner":

    st.subheader("🏆 Constituency Winner Result")

    constituency = st.selectbox(
        "Select Constituency",
        sorted(df['Constituency'].unique())
    )

    const_data = df[
        df['Constituency'] == constituency
    ]

    winner = const_data.loc[
        const_data['Total Votes'].idxmax()
    ]

    st.markdown(f"""
    <div class='winner-box'>
    🎉 {winner['Candidate']} WON THE ELECTION 🎉
    </div>
    """, unsafe_allow_html=True)

    st.balloons()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Party", winner['Party'])

    with col2:
        st.metric("Votes", int(winner['Total Votes']))

    with col3:
        st.metric("Vote %", winner['% Votes'])

    ranked = const_data.sort_values(
        by='Total Votes',
        ascending=False
    )

    st.subheader("📋 Candidate Ranking")

    st.dataframe(
        ranked[['Candidate','Party','Total Votes','% Votes']]
    )

# =========================================================
# PREDICTION
# =========================================================

elif menu == "🤖 Election Prediction":

    st.subheader("🤖 AI Election Prediction")

    col1, col2 = st.columns(2)

    with col1:

        evm_votes = st.number_input(
            "EVM Votes",
            min_value=0
        )

        postal_votes = st.number_input(
            "Postal Votes",
            min_value=0
        )

        total_votes = st.number_input(
            "Total Votes",
            min_value=0
        )

    with col2:

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

    if st.button("Predict Election Result"):

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

            st.snow()

            st.markdown("""
            <div class='winner-box'>
            🏆 HIGH WINNING CHANCE 🏆
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(probability))

            st.success(
                f"Winning Probability: {round(probability*100,2)}%"
            )

        else:

            st.markdown("""
            <div class='loser-box'>
            ❌ LOW WINNING CHANCE
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(probability))

            st.error(
                f"Winning Probability: {round(probability*100,2)}%"
            )

# =========================================================
# VOTE SHARE
# =========================================================

elif menu == "📈 Vote Share":

    st.subheader("📈 Party Vote Share")

    seats = df[df['Winner'] == 1]['Party'].value_counts()

    fig2 = px.pie(
        names=seats.index,
        values=seats.values,
        hole=0.4
    )

    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<hr>

<center>

<h4 style='color:white'>

🗳️ Tamil Nadu Election AI Dashboard

</h4>

<p style='color:lightgray'>

Machine Learning • Streamlit • XGBoost • Election Analytics

</p>

</center>
""", unsafe_allow_html=True)
