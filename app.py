# ==========================================
# STANDARD ELECTION STREAMLIT UI
# CLEAN + PROFESSIONAL DESIGN
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
    page_title="Tamil Nadu Election Prediction",
    page_icon="🗳️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background-image: linear-gradient(
    rgba(0,0,0,0.75),
    rgba(0,0,0,0.75)),
    url('https://images.unsplash.com/photo-1529107386315-e1a2ed48a620?q=80&w=2070');
    
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #FFD700;
    padding: 10px;
}

/* Card Style */
.card {
    background-color: rgba(255,255,255,0.12);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 0px 20px rgba(255,255,255,0.2);
    backdrop-filter: blur(6px);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.85);
}

/* Text */
h1,h2,h3,h4,p,label {
    color: white !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 25px;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(90deg,#11998e,#38ef7d);
    color: white;
}

/* Metric */
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.12);
    border-radius: 15px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown(
    "<div class='main-title'>Tamil Nadu Election Prediction</div>",
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

# Create Winner Column

max_votes = df.groupby('Constituency')['Total Votes'].transform('max')

df['Winner'] = np.where(
    df['Total Votes'] == max_votes,
    1,
    0
)

# Encode Data

party_encoder = LabelEncoder()
df['Party_Encoded'] = party_encoder.fit_transform(df['Party'])

const_encoder = LabelEncoder()
df['Constituency_Encoded'] = const_encoder.fit_transform(df['Constituency'])

# ---------------- MODEL ----------------

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

# ---------------- SIDEBAR ----------------

st.sidebar.title("Dashboard")

menu = st.sidebar.radio(
    "Select",
    [
        "Home",
        "Dataset",
        "Analytics",
        "Constituency Result",
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

    <h2>About Project</h2>

    This dashboard analyzes Tamil Nadu election data
    using Machine Learning models for election prediction
    and constituency analysis.

    Models Used:
    
    • Logistic Regression  
    • Random Forest  
    • XGBoost  

    </div>
    """, unsafe_allow_html=True)

# ================= DATASET =================

elif menu == "Dataset":

    st.subheader("Election Dataset")

    st.dataframe(df)

# ================= ANALYTICS =================

elif menu == "Analytics":

    st.subheader("Election Analytics")

    # Top Parties

    party_votes = df.groupby('Party')['Total Votes'] \
                    .sum() \
                    .sort_values(ascending=False) \
                    .head(10)

    fig = px.bar(
        x=party_votes.index,
        y=party_votes.values,
        labels={
            'x':'Party',
            'y':'Votes'
        },
        title="Top Parties by Total Votes"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Winners Count

    winners = df[df['Winner'] == 1]['Party'].value_counts()

    fig2 = px.pie(
        names=winners.index,
        values=winners.values,
        title="Winning Seat Share"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ================= CONSTITUENCY RESULT =================

elif menu == "Constituency Result":

    st.subheader("Constituency Winner")

    constituency = st.selectbox(
        "Select Constituency",
        sorted(df['Constituency'].unique())
    )

    const_data = df[df['Constituency'] == constituency]

    winner = const_data.loc[
        const_data['Total Votes'].idxmax()
    ]

    st.success(
        f"Winning Candidate: {winner['Candidate']}"
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
            "Vote Percentage",
            winner['% Votes']
        )

    st.subheader("Candidate Ranking")

    ranked = const_data.sort_values(
        by='Total Votes',
        ascending=False
    )

    st.dataframe(
        ranked[['Candidate','Party','Total Votes']]
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

            st.success("Predicted Result : Winner")

        else:

            st.error("Predicted Result : Not Winner")

        st.progress(float(probability))

        st.write(
            f"Winning Probability : {round(probability*100,2)}%"
        )

# ---------------- FOOTER ----------------

st.markdown("""
<hr>

<center>

<p style='color:white'>
Tamil Nadu Election Prediction Dashboard
</p>

</center>
""", unsafe_allow_html=True)
