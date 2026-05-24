# ==============================
# STREAMLIT APP - Election Winner Prediction
# ==============================

# Save this file as:
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="Tamil Nadu Election Prediction",
    page_icon="🗳️",
    layout="wide"
)

# ------------------------------
# CUSTOM BACKGROUND + DESIGN
# ------------------------------

page_bg = """
<style>

[data-testid="stAppViewContainer"]{
background: linear-gradient(to right, #141e30, #243b55);
color: white;
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

.big-title{
font-size:55px;
font-weight:bold;
text-align:center;
color:white;
padding-top:20px;
}

.sub-title{
font-size:22px;
text-align:center;
color:#d1d1d1;
margin-bottom:40px;
}

.result-win{
background-color:#00c853;
padding:20px;
border-radius:15px;
text-align:center;
font-size:30px;
font-weight:bold;
color:white;
}

.result-lose{
background-color:#d50000;
padding:20px;
border-radius:15px;
text-align:center;
font-size:30px;
font-weight:bold;
color:white;
}

.stButton>button{
width:100%;
background-color:#ff9800;
color:white;
font-size:20px;
font-weight:bold;
border-radius:10px;
height:3em;
border:none;
}

.stButton>button:hover{
background-color:#ff5722;
color:white;
}

.css-1d391kg{
background-color:#111111;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ------------------------------
# LOAD MODEL
# ------------------------------

model = joblib.load("tamilnadu_election_model.pkl")

# ------------------------------
# TITLE
# ------------------------------

st.markdown(
    '<p class="big-title">🗳 Tamil Nadu Election Winner Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Machine Learning Based Election Prediction System</p>',
    unsafe_allow_html=True
)

# ------------------------------
# SIDEBAR
# ------------------------------

st.sidebar.title("📌 Input Election Details")

evm_votes = st.sidebar.number_input(
    "EVM Votes",
    min_value=0,
    value=70000
)

postal_votes = st.sidebar.number_input(
    "Postal Votes",
    min_value=0,
    value=500
)

total_votes = st.sidebar.number_input(
    "Total Votes",
    min_value=0,
    value=70500
)

vote_percent = st.sidebar.slider(
    "% Votes",
    0.0,
    100.0,
    40.2
)

party_encoded = st.sidebar.number_input(
    "Party Encoded Value",
    min_value=0,
    value=86
)

constituency_encoded = st.sidebar.number_input(
    "Constituency Encoded Value",
    min_value=0,
    value=3
)

# ------------------------------
# MAIN CONTENT
# ------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("📊 Candidate Input Data")

    input_df = pd.DataFrame({
        'EVM Votes': [evm_votes],
        'Postal Votes': [postal_votes],
        'Total Votes': [total_votes],
        '% Votes': [vote_percent],
        'Party_Encoded': [party_encoded],
        'Constituency_Encoded': [constituency_encoded]
    })

    st.dataframe(input_df, use_container_width=True)

with col2:

    st.subheader("🤖 ML Prediction")

    if st.button("Predict Winner"):

        prediction = model.predict(input_df)

        if prediction[0] == 1:

            st.markdown(
                '<div class="result-win">🏆 Predicted Result: WINNER</div>',
                unsafe_allow_html=True
            )

            st.balloons()

        else:

            st.markdown(
                '<div class="result-lose">❌ Predicted Result: NOT WINNER</div>',
                unsafe_allow_html=True
            )

# ------------------------------
# FOOTER
# ------------------------------

st.markdown("---")

st.markdown(
    """
    <center>
    <h4>Developed using Streamlit, Machine Learning & XGBoost</h4>
    </center>
    """,
    unsafe_allow_html=True
)
