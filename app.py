import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

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
    "<h1 class='main-title'>🗳️ Tamil Nadu Election Prediction</h1>",
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("eci_results_tamilnadu_2026.csv")
    return df

df = load_data()

# ---------------- DATA CLEANING ----------------
df = df.dropna()

# Winner column
max_votes = df.groupby('Constituency')['Total Votes'].transform('max')
df['Winner'] = np.where(df['Total Votes'] == max_votes, 1, 0)

# Encode Party
party_encoder = LabelEncoder()
df['Party_Encoded'] = party_encoder.fit_transform(df['Party'])

# Encode Constituency
const_encoder = LabelEncoder()
df['Constituency_Encoded'] = const_encoder.fit_transform(df['Constituency'])

# ---------------- SIDEBAR ----------------
st.sidebar.title(" Dashboard Menu")

menu = st.sidebar.radio(
    "Choose Option",
    [
        "Home",
        "Dataset",
        "Top Parties",
        "Constituency Winner",
        "Election Prediction",
        "Analytics"
    ]
)

# ---------------- HOME ----------------
if menu == "Home":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Constituencies", df['Constituency'].nunique())

    with col2:
        st.metric("Total Candidates", df['Candidate'].nunique())

    with col3:
        st.metric("Total Parties", df['Party'].nunique())

    st.markdown("<br>", unsafe_allow_html=True)


# ---------------- DATASET ----------------
elif menu == "Dataset":

    st.subheader("📄 Election Dataset")

    st.dataframe(df)

# ---------------- TOP PARTIES ----------------
elif menu == "Top Parties":

    st.subheader("🏆 Top Parties by Votes")

    party_votes = df.groupby('Party')['Total Votes'].sum().sort_values(ascending=False).head(10)

    fig = px.bar(
        x=party_votes.index,
        y=party_votes.values,
        labels={'x':'Party','y':'Votes'},
        title='Top 10 Parties'
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- CONSTITUENCY WINNER ----------------
elif menu == "Constituency Winner":

    st.subheader("🎯 Find Constituency Winner")

    constituency = st.selectbox(
        "Select Constituency",
        sorted(df['Constituency'].unique())
    )

    const_data = df[df['Constituency'] == constituency]

    winner = const_data.loc[const_data['Total Votes'].idxmax()]

    st.success(f"🏆 Winner: {winner['Candidate']}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Party", winner['Party'])

    with col2:
        st.metric("Votes", int(winner['Total Votes']))

    with col3:
        st.metric("Vote %", winner['% Votes'])

    # Ranking Table
    ranked = const_data.sort_values(by='Total Votes', ascending=False)

    st.subheader("📊 Candidate Rankings")

    st.dataframe(
        ranked[['Candidate','Party','Total Votes','% Votes']]
    )

# ---------------- PREDICTION ----------------
elif menu == "Election Prediction":

    st.subheader("🤖 Predict Election Winner")

    evm_votes = st.number_input("EVM Votes", min_value=0)

    postal_votes = st.number_input("Postal Votes", min_value=0)

    total_votes = st.number_input("Total Votes", min_value=0)

    percent_votes = st.slider("% Votes", 0.0, 100.0, 40.0)

    party = st.selectbox(
        "Select Party",
        sorted(df['Party'].unique())
    )

    constituency = st.selectbox(
        "Select Constituency",
        sorted(df['Constituency'].unique())
    )

    # Encode values
    party_encoded = party_encoder.transform([party])[0]
    const_encoded = const_encoder.transform([constituency])[0]

    # Load model
    model = joblib.load("tamilnadu_election_model.pkl")

    if st.button("Predict Result"):

        sample = pd.DataFrame({
            'EVM Votes': [evm_votes],
            'Postal Votes': [postal_votes],
            'Total Votes': [total_votes],
            '% Votes': [percent_votes],
            'Party_Encoded': [party_encoded],
            'Constituency_Encoded': [const_encoded]
        })

        prediction = model.predict(sample)

        if prediction[0] == 1:
            st.balloons()

            st.success("🏆 Predicted Result: WINNER")

        else:
            st.error("❌ Predicted Result: NOT WINNER")

# ---------------- ANALYTICS ----------------
elif menu == "Analytics":

    st.subheader("📈 Election Analytics")

    # Seats count
    party_seats = df[df['Winner'] == 1]['Party'].value_counts().head(10)

    fig1 = px.pie(
        names=party_seats.index,
        values=party_seats.values,
        title="Seat Share"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Vote Share
    vote_share = df.groupby('Party')['Total Votes'].sum().sort_values(ascending=False).head(10)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=vote_share.index,
        y=vote_share.values,
        mode='lines+markers'
    ))

    fig2.update_layout(
        title="Vote Share Trend",
        xaxis_title="Party",
        yaxis_title="Votes"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ---------------- FOOTER ----------------
