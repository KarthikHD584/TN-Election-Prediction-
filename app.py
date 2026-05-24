# =========================================================
# TAMIL NADU ELECTION PREDICTION - ADVANCED STREAMLIT APP
# =========================================================

# SAVE AS: app.py

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TN Election Prediction",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f2027,
        #203a43,
        #2c5364
    );
    color: white;
}

/* Header */
.main-title {
    font-size: 55px;
    font-weight: 700;
    text-align: center;
    color: #ffffff;
    margin-top: 10px;
}

.subtitle {
    font-size: 20px;
    text-align: center;
    color: #d3d3d3;
    margin-bottom: 40px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
}

/* Prediction Winner */
.winner {
    background: linear-gradient(to right, #11998e, #38ef7d);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 35px;
    font-weight: bold;
    color: white;
}

/* Prediction Loser */
.loser {
    background: linear-gradient(to right, #cb2d3e, #ef473a);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 35px;
    font-weight: bold;
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #ff512f, #dd2476);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    height: 3.2em;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(to right, #24c6dc, #514a9d);
    color: white;
}

/* Metrics */
.metric-box {
    background: rgba(255,255,255,0.07);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

model = load("tamilnadu_election_model.pkl")

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🗳️ Tamil Nadu Election Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Election Winner Prediction Dashboard</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.title("📌 Candidate Details")

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
    "Vote Percentage",
    0.0,
    100.0,
    40.2
)

party_encoded = st.sidebar.number_input(
    "Party Encoded",
    min_value=0,
    value=86
)

constituency_encoded = st.sidebar.number_input(
    "Constituency Encoded",
    min_value=0,
    value=3
)

# =========================================================
# INPUT DATAFRAME
# =========================================================

input_df = pd.DataFrame({
    'EVM Votes': [evm_votes],
    'Postal Votes': [postal_votes],
    'Total Votes': [total_votes],
    '% Votes': [vote_percent],
    'Party_Encoded': [party_encoded],
    'Constituency_Encoded': [constituency_encoded]
})

# =========================================================
# DASHBOARD LAYOUT
# =========================================================

col1, col2 = st.columns([1,1])

# =========================================================
# LEFT SIDE
# =========================================================

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Candidate Data")

    st.dataframe(
        input_df,
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # Gauge Chart

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = vote_percent,
        title = {'text': "Vote Percentage"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "lime"},
            'steps': [
                {'range': [0, 35], 'color': "red"},
                {'range': [35, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "green"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# RIGHT SIDE
# =========================================================

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🤖 Prediction System")

    if st.button("Predict Election Winner"):

        prediction = model.predict(input_df)

        if prediction[0] == 1:

            st.markdown(
                '<div class="winner">🏆 WINNER PREDICTED</div>',
                unsafe_allow_html=True
            )

            st.balloons()

        else:

            st.markdown(
                '<div class="loser">❌ NOT A WINNER</div>',
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # Pie Chart

    pie_df = pd.DataFrame({
        'Category': ['EVM Votes', 'Postal Votes'],
        'Votes': [evm_votes, postal_votes]
    })

    pie_chart = px.pie(
        pie_df,
        names='Category',
        values='Votes',
        hole=0.5
    )

    st.plotly_chart(pie_chart, use_container_width=True)

# =========================================================
# METRICS
# =========================================================

st.markdown("## 📈 Election Metrics")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        label="Total Votes",
        value=f"{total_votes:,}"
    )

with m2:
    st.metric(
        label="Vote %",
        value=f"{vote_percent}%"
    )

with m3:
    st.metric(
        label="Postal Votes",
        value=f"{postal_votes:,}"
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<center>
<h4>Developed with ❤️ using Streamlit, XGBoost & Machine Learning</h4>
</center>
""", unsafe_allow_html=True)
