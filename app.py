# ==========================================
# TAMIL NADU ELECTION PREDICTION
# STANDARD PROFESSIONAL UI
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="TAMIL NADU ELECTION PREDICTION",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main Background */

.stApp {
   background-image:
    linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.82)),
    url("https://images.unsplash.com/photo-1529107386315-e1a2ed48a620?q=80&w=2070");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: rgba(15,15,15,0.95);
}

/* Title */

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 10px;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #dcdcdc;
    margin-bottom: 30px;
}

/* Cards */

.card {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Metrics */

[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px;
}

/* Text */

h1,h2,h3,h4,h5,h6,p,label,span {
    color: white !important;
}

/* Buttons */

.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 22px;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #0d5ea8;
    color: white;
}

/* Table */

[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.04);
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown("""
<div class="main-title">
Tamil Nadu Election Prediction Dashboard
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    df = pd.read_csv("eci_results_tamilnadu_2026.csv")
    return df

df = load_data()

df = df.dropna()

# ---------------- CREATE WINNER COLUMN ----------------

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

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select",
    [
        "Home",
        "Election Analytics",
        "Constituency Result",
        "Winner Prediction",
        "Winning Parties"
    ]
)

# ---------------- HOME ----------------

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
            "Political Parties",
            df['Party'].nunique()
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">

    <h3>About Project</h3>

    This dashboard analyzes Tamil Nadu election data
    using Machine Learning models and predicts election winners.

    Models Used:
    
    Logistic Regression  
    Random Forest  
    XGBoost  

    </div>
    """, unsafe_allow_html=True)

# ---------------- ANALYTICS ----------------

elif menu == "Election Analytics":

    st.subheader("Top Parties by Votes")

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
        }
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- CONSTITUENCY RESULT ----------------

elif menu == "Constituency Result":

    st.subheader("Constituency Winner")

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
        f"Winning Candidate: {winner['Candidate']}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Party", winner['Party'])

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

    ranked = const_data.sort_values(
        by='Total Votes',
        ascending=False
    )

    st.dataframe(
        ranked[
            ['Candidate','Party','Total Votes']
        ]
    )

# ---------------- WINNER PREDICTION ----------------

elif menu == "Winner Prediction":

    st.subheader("Predict Election Winner")

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

    percent_votes = st.slider(
        "Vote Percentage",
        0.0,
        100.0,
        40.0
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
            '% Votes': [percent_votes],
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

# ---------------- WINNING PARTIES ----------------

elif menu == "Winning Parties":

    st.subheader("Winning Political Parties")

    party_seats = df[df['Winner'] == 1]['Party'].value_counts()

    result_df = pd.DataFrame({
        'Party': party_seats.index,
        'Seats Won': party_seats.values
    })

    # Add Party Symbols

    symbols = {

        "DMK": "☀️ Rising Sun",
        "AIADMK": "🍃 Two Leaves",
        "BJP": "🌸 Lotus",
        "INC": "✋ Hand",
        "PMK": "🥭 Mango",
        "NTK": "🎤 Microphone",
        "CPI": "🌾 Ears of Corn",
        "CPI(M)": "🔨 Hammer Sickle"

    }

    result_df['Symbol'] = result_df['Party'].map(symbols)

    st.dataframe(result_df)

    fig2 = px.pie(
        result_df,
        names='Party',
        values='Seats Won'
    )

    st.plotly_chart(fig2, use_container_width=True)

# ---------------- FOOTER ----------------

st.markdown("""
<hr>

<center>

Tamil Nadu Election Prediction System

</center>
""", unsafe_allow_html=True)
