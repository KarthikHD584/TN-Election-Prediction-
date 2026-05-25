# ==========================================
# 🗳️ TAMIL NADU ELECTION PREDICTION
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title=" TAMIL NADU ELECTION PREDICTION",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS - PREMIUM 3D DESIGN
# ==========================================

st.markdown("""
<style>

/* IMPORT FONT */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* MAIN BACKGROUND */

.stApp {
    background:
    linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.88)),
    url("https://images.unsplash.com/photo-1523995462485-3d171b5c8fa9?q=80&w=2070");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* HIDE STREAMLIT MENU */

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* MAIN TITLE */

.main-title {
    text-align: center;
    font-size: 70px;
    font-weight: 900;

    background: linear-gradient(
        90deg,
        #ffcc00,
        #ff6600,
        #ff0000,
        #ffcc00
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    letter-spacing: 2px;

    text-shadow:
        0px 0px 15px rgba(255,255,255,0.3);

    animation: glow 3s infinite alternate;
}

/* TITLE GLOW */

@keyframes glow {
    from {
        filter: drop-shadow(0px 0px 10px gold);
    }
    to {
        filter: drop-shadow(0px 0px 25px orange);
    }
}

/* SUBTITLE */

.sub-title {
    text-align: center;
    font-size: 24px;
    color: #ffffff;
    margin-top: -10px;
    margin-bottom: 40px;
    font-weight: 300;
}

/* GLASS CARD */

.glass {
    background: rgba(255,255,255,0.08);

    border-radius: 25px;

    padding: 30px;

    backdrop-filter: blur(12px);

    border: 1px solid rgba(255,255,255,0.15);

    box-shadow:
        0px 10px 30px rgba(0,0,0,0.5),
        inset 0px 0px 10px rgba(255,255,255,0.1);

    transition: 0.4s;
}

.glass:hover {
    transform: translateY(-8px) scale(1.02);

    box-shadow:
        0px 20px 40px rgba(255,140,0,0.4);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        rgba(10,10,10,0.98),
        rgba(25,25,25,0.98)
    );

    border-right: 2px solid rgba(255,255,255,0.08);
}

/* SIDEBAR TITLE */

.sidebar-title {
    color: gold;
    font-size: 28px;
    text-align: center;
    font-weight: bold;
}

/* METRIC CARDS */

[data-testid="metric-container"] {

    background:
    linear-gradient(
        135deg,
        rgba(255,255,255,0.12),
        rgba(255,255,255,0.05)
    );

    border-radius: 20px;

    padding: 25px;

    border: 1px solid rgba(255,255,255,0.12);

    box-shadow:
        0px 10px 20px rgba(0,0,0,0.3);

    transition: 0.3s;
}

[data-testid="metric-container"]:hover {

    transform: scale(1.05);

    border: 1px solid gold;

    box-shadow:
        0px 0px 20px rgba(255,215,0,0.5);
}

/* METRIC TEXT */

[data-testid="metric-container"] label {
    color: #FFD700 !important;
    font-size: 18px !important;
}

[data-testid="metric-container"] div {
    color: white !important;
}

/* BUTTON */

.stButton>button {

    width: 100%;

    background:
    linear-gradient(
        90deg,
        #ff512f,
        #dd2476
    );

    color: white;

    border: none;

    border-radius: 15px;

    padding: 14px;

    font-size: 20px;

    font-weight: bold;

    transition: 0.4s;

    box-shadow:
        0px 8px 20px rgba(255,0,128,0.4);
}

.stButton>button:hover {

    transform: scale(1.03);

    background:
    linear-gradient(
        90deg,
        #00c6ff,
        #0072ff
    );

    box-shadow:
        0px 10px 25px rgba(0,114,255,0.6);
}

/* SELECT BOX */

.stSelectbox div[data-baseweb="select"] {

    background-color: rgba(255,255,255,0.1);

    border-radius: 12px;
}

/* NUMBER INPUT */

.stNumberInput input {

    background-color: rgba(255,255,255,0.1);

    color: white;

    border-radius: 12px;
}

/* SLIDER */

.stSlider {

    color: gold;
}

/* TABLE */

[data-testid="stDataFrame"] {

    background: rgba(255,255,255,0.05);

    border-radius: 15px;

    padding: 10px;
}

/* HEADINGS */

h1, h2, h3, h4, h5, h6 {

    color: #FFD700 !important;

    font-weight: 700;
}

/* TEXT */

p, label, span, div {

    color: white;
}

/* WINNER CELEBRATION */

.winner-box {

    text-align: center;

    background:
    linear-gradient(
        135deg,
        rgba(255,215,0,0.15),
        rgba(255,140,0,0.18)
    );

    padding: 35px;

    border-radius: 30px;

    border: 2px solid gold;

    animation: pulse 2s infinite;

    box-shadow:
        0px 0px 30px rgba(255,215,0,0.4);
}

/* PULSE EFFECT */

@keyframes pulse {

    0% {
        transform: scale(1);
    }

    50% {
        transform: scale(1.02);
    }

    100% {
        transform: scale(1);
    }
}

/* FOOTER */

.footer {
    text-align: center;
    margin-top: 40px;
    color: silver;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.markdown("""
<div class="main-title">
🗳️Election Prediction
</div>
""", unsafe_allow_html=True)


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("eci_results_tamilnadu_2026.csv")

df = load_data()

df = df.dropna()

# ==========================================
# WINNER COLUMN
# ==========================================

max_votes = df.groupby('Constituency')['Total Votes'].transform('max')

df['Winner'] = np.where(
    df['Total Votes'] == max_votes,
    1,
    0
)

# ==========================================
# ENCODING
# ==========================================

party_encoder = LabelEncoder()
df['Party_Encoded'] = party_encoder.fit_transform(df['Party'])

const_encoder = LabelEncoder()
df['Constituency_Encoded'] = const_encoder.fit_transform(df['Constituency'])

# ==========================================
# FEATURES
# ==========================================

X = df[[
    'EVM Votes',
    'Postal Votes',
    'Total Votes',
    '% Votes',
    'Party_Encoded',
    'Constituency_Encoded'
]]

y = df['Winner']

# ==========================================
# TRAIN MODEL
# ==========================================

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

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown("""
<div class="sidebar-title">
🗳️ Navigation
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    [
        " Home",
        " Dataset",
        " Analytics",
        " Constituency Winner",
        " Election Prediction",
        " Sentiment Analysis"
    ]
)

# ==========================================
# HOME
# ==========================================

if menu == " Home":

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


# ==========================================
# DATASET
# ==========================================

elif menu == " Dataset":

    st.subheader(" Tamil Nadu Election Dataset")

    st.dataframe(df, use_container_width=True)

# ==========================================
# ANALYTICS
# ==========================================

elif menu == " Analytics":

    st.subheader(" Election Analytics")

    party_votes = df.groupby('Party')['Total Votes'] \
                    .sum() \
                    .sort_values(ascending=False) \
                    .head(10)

    fig = px.bar(
        x=party_votes.index,
        y=party_votes.values,
        color=party_votes.values,
        text=party_votes.values,
        title="Top Parties by Votes"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # PIE CHART

    party_seats = df[df['Winner'] == 1]['Party'].value_counts()

    fig2 = px.pie(
        names=party_seats.index,
        values=party_seats.values,
        hole=0.45,
        title="Seat Share"
    )

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# WINNER
# ==========================================

elif menu == " Constituency Winner":

    st.subheader("🏆 Constituency Winner")

    constituency = st.selectbox(
        "Select Constituency",
        sorted(df['Constituency'].unique())
    )

    const_data = df[df['Constituency'] == constituency]

    winner = const_data.loc[
        const_data['Total Votes'].idxmax()
    ]

    st.markdown(f"""
    <div class="winner-box">

    </div>
    """, unsafe_allow_html=True)

    st.balloons()

    st.markdown("<br>", unsafe_allow_html=True)

    ranked = const_data.sort_values(
        by='Total Votes',
        ascending=False
    )

    st.subheader("📋 Candidate Ranking")

    st.dataframe(
        ranked[
            ['Candidate', 'Party', 'Total Votes', '% Votes']
        ],
        use_container_width=True
    )

# ==========================================
# PREDICTION
# ==========================================

elif menu == "🔮 Election Prediction":

    st.subheader("🔮 Predict Future Election Winner")

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

    if st.button("🚀 Predict Winner"):

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

            st.markdown(f"""
            <div class="winner-box">

                <h1>🎊 PARTY CELEBRATION 🎊</h1>

                <h1>🏆 WINNER PREDICTED</h1>

                <h2>{party}</h2>

                <h2>🔥 Winning Probability</h2>

                <h1>{round(probability*100,2)}%</h1>

            </div>
            """, unsafe_allow_html=True)

            st.progress(float(probability))

        else:

            st.error("❌ Predicted Result: NOT WINNER")

            st.progress(float(probability))

            st.write(
                f"Winning Probability: {round(probability*100,2)}%"
            )

# ==========================================
# SENTIMENT ANALYSIS
# ==========================================

elif menu == "📈 Sentiment Analysis":

    st.subheader("📈 Political Sentiment Analysis")

    sentiment_data = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Count': [65, 20, 15]
    })

    fig = px.bar(
        sentiment_data,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        text='Count',
        title='Public Sentiment Analysis'
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# FOOTER
# ==========================================

