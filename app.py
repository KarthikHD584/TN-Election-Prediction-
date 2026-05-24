# =========================================
# TAMIL NADU ELECTION PREDICTION DASHBOARD
# Professional Streamlit UI Design
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Election Prediction Dashboard",
    page_icon="🗳️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background-image:
    linear-gradient(rgba(0,0,0,0.75),
    rgba(0,0,0,0.75)),
    url("https://images.unsplash.com/photo-1541872703-74c5e44368f9?q=80&w=2070");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.main-title {
    text-align: center;
    font-size: 50px;
    color: white;
    font-weight: bold;
    margin-top: 10px;
}

.sub-title {
    text-align: center;
    color: #dcdcdc;
    font-size: 20px;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(5px);
}

section[data-testid="stSidebar"] {
    background-color: rgba(0,0,0,0.9);
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 15px;
}

h1,h2,h3,h4,h5,p,label {
    color: white !important;
}

.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 22px;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #0d5ea8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown(
    "<div class='main-title'>Tamil Nadu Election Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Machine Learning Based Election Analysis System</div>",
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv("eci_results_tamilnadu_2026.csv")

df = load_data()

# ---------------- DATA CLEANING ----------------

df = df.dropna()

max_votes = df.groupby('Constituency')['Total Votes'].transform('max')

df['Winner'] = np.where(
    df['Total Votes'] == max_votes,
    1,
    0
)

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

# ---------------- MODEL TRAINING ----------------

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

joblib.dump(model, "election_model.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.title("Dashboard")

menu = st.sidebar.selectbox(
    "Select Option",
    [
        "Home",
        "Dataset",
        "Analytics",
        "Constituency Winner",
        "Prediction"
    ]
)

# ================= HOME =================

if menu == "Home":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Constituencies",
            df['Constituency'].nunique()
        )

    with col2:
        st.metric(
            "Candidates",
            df['Candidate'].nunique()
        )

    with col3:
        st.metric(
            "Parties",
            df['Party'].nunique()
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    <h2>Project Overview</h2>

    This dashboard predicts election winners using
    machine learning algorithms and election data.

    Features included:

    • Winner prediction  
    • Constituency analysis  
    • Party vote comparison  
    • Interactive charts  
    • Election analytics  

    </div>
    """, unsafe_allow_html=True)

# ================= DATASET =================

elif menu == "Dataset":

    st.subheader("Election Dataset")

    st.dataframe(df)

# ================= ANALYTICS =================

elif menu == "Analytics":

    st.subheader("Party Vote Analysis")

    party_votes = df.groupby('Party')['Total Votes'] \
                    .sum() \
                    .sort_values(ascending=False) \
                    .head(10)

    fig = px.bar(
        x=party_votes.index,
        y=party_votes.values,
        labels={
            'x': 'Party',
            'y': 'Votes'
        },
        title="Top 10 Parties"
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= WINNER =================

elif menu == "Constituency Winner":

    st.subheader("Find Constituency Winner")

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

    st.success(
        f"Winner: {winner['Candidate']}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Party",
            winner['Party']
        )

    with col2:
        st.metric(
            "Votes",
            int(winner['Total Votes'])
        )

    with col3:
        st.metric(
            "Vote %",
            winner['% Votes']
        )

    st.subheader("Candidate Ranking")

    ranked = const_data.sort_values(
        by='Total Votes',
        ascending=False
    )

    st.dataframe(
        ranked[
            [
                'Candidate',
                'Party',
                'Total Votes',
                '% Votes'
            ]
        ]
    )

# ================= PREDICTION =================

elif menu == "Prediction":

    st.subheader("Election Winner Prediction")

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

    vote_percent = st.slider(
        "Vote Percentage",
        0.0,
        100.0,
        45.0
    )

    party = st.selectbox(
        "Party",
        sorted(df['Party'].unique())
    )

    constituency = st.selectbox(
        "Constituency",
        sorted(df['Constituency'].unique())
    )

    party_encoded = party_encoder.transform([party])[0]

    const_encoded = const_encoder.transform([constituency])[0]

    if st.button("Predict"):

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

            st.success("Predicted Result: Winner")

        else:

            st.error("Predicted Result: Not Winner")

        st.progress(float(probability))

        st.write(
            f"Winning Probability: {round(probability*100,2)}%"
        )

# ---------------- FOOTER ----------------

st.markdown("""
<hr>

<center>

<p style='color:white'>
Election Prediction Dashboard using Streamlit
</p>

</center>
""", unsafe_allow_html=True)
