# =========================================================
# TAMIL NADU ELECTION WINNER PREDICTION - ADVANCED ML MODEL
# BEST STANDARD ML PREDICTION CODE
# =========================================================

# INSTALL REQUIRED LIBRARIES:
# pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn joblib

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("eci_results_tamilnadu_2026.csv")

print("\n========== DATASET HEAD ==========\n")
print(df.head())

print("\n========== DATASET INFO ==========\n")
print(df.info())

print("\n========== NULL VALUES ==========\n")
print(df.isnull().sum())

# =========================================================
# DATA CLEANING
# =========================================================

df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# =========================================================
# CREATE TARGET COLUMN (WINNER)
# =========================================================

max_votes = df.groupby('Constituency')['Total Votes'].transform('max')

df['Winner'] = np.where(df['Total Votes'] == max_votes, 1, 0)

print("\n========== WINNER COLUMN CREATED ==========\n")
print(df[['Constituency', 'Candidate', 'Party', 'Total Votes', 'Winner']].head())

# =========================================================
# LABEL ENCODING
# =========================================================

party_encoder = LabelEncoder()
candidate_encoder = LabelEncoder()
const_encoder = LabelEncoder()

df['Party_Encoded'] = party_encoder.fit_transform(df['Party'])

df['Candidate_Encoded'] = candidate_encoder.fit_transform(df['Candidate'])

df['Constituency_Encoded'] = const_encoder.fit_transform(df['Constituency'])

# =========================================================
# FEATURE ENGINEERING
# =========================================================

# Vote Share Ratio
df['Vote_Share'] = df['Total Votes'] / df.groupby('Constituency')['Total Votes'].transform('sum')

# Margin Estimate
df['Vote_Strength'] = df['EVM Votes'] + df['Postal Votes']

# =========================================================
# SELECT FEATURES
# =========================================================

X = df[[
    'EVM Votes',
    'Postal Votes',
    'Total Votes',
    '% Votes',
    'Vote_Share',
    'Vote_Strength',
    'Party_Encoded',
    'Candidate_Encoded',
    'Constituency_Encoded'
]]

y = df['Winner']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# =========================================================
# MODEL 1 : LOGISTIC REGRESSION
# =========================================================

lr_model = LogisticRegression(max_iter=5000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)

print("\n========== LOGISTIC REGRESSION ==========")
print("Accuracy:", lr_acc)

# =========================================================
# MODEL 2 : RANDOM FOREST
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("\n========== RANDOM FOREST ==========")
print("Accuracy:", rf_acc)

# =========================================================
# MODEL 3 : XGBOOST (BEST MODEL)
# =========================================================

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_pred)

print("\n========== XGBOOST ==========")
print("Accuracy:", xgb_acc)

# =========================================================
# BEST MODEL SELECTION
# =========================================================

accuracies = {
    "Logistic Regression": lr_acc,
    "Random Forest": rf_acc,
    "XGBoost": xgb_acc
}

best_model_name = max(accuracies, key=accuracies.get)

print("\n========== BEST MODEL ==========")
print("Best Model:", best_model_name)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(classification_report(y_test, xgb_pred))

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, xgb_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========\n")
print(importance)

# Plot Feature Importance
plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.title("Feature Importance - XGBoost")

plt.show()

# =========================================================
# MODEL COMPARISON GRAPH
# =========================================================

models = list(accuracies.keys())
scores = list(accuracies.values())

plt.figure(figsize=(8,5))

bars = plt.bar(models, scores)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + 0.1,
        yval + 0.005,
        round(yval, 4)
    )

plt.show()

# =========================================================
# PARTY PERFORMANCE ANALYSIS
# =========================================================

party_votes = df.groupby('Party')['Total Votes'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))

party_votes.plot(kind='bar')

plt.title("Top 10 Parties by Total Votes")
plt.xlabel("Party")
plt.ylabel("Votes")

plt.show()

# =========================================================
# FUTURE ELECTION PREDICTION
# =========================================================

print("\n========== FUTURE ELECTION PREDICTION ==========\n")

# Example Prediction Input
party_name = "DMK"
candidate_name = "Candidate A"
constituency_name = "Chennai Central"

# Encode Values
party_encoded = party_encoder.transform([party_name])[0]
candidate_encoded = 0
const_encoded = 0

sample_data = pd.DataFrame({
    'EVM Votes': [80000],
    'Postal Votes': [1000],
    'Total Votes': [81000],
    '% Votes': [48.5],
    'Vote_Share': [0.48],
    'Vote_Strength': [81000],
    'Party_Encoded': [party_encoded],
    'Candidate_Encoded': [candidate_encoded],
    'Constituency_Encoded': [const_encoded]
})

prediction = xgb_model.predict(sample_data)

probability = xgb_model.predict_proba(sample_data)[0][1]

if prediction[0] == 1:
    print("🏆 Predicted Result: WINNER")
else:
    print("❌ Predicted Result: NOT WINNER")

print(f"Winning Probability: {probability*100:.2f}%")

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(xgb_model, "tamilnadu_best_model.pkl")

joblib.dump(party_encoder, "party_encoder.pkl")

joblib.dump(candidate_encoder, "candidate_encoder.pkl")

joblib.dump(const_encoder, "constituency_encoder.pkl")

print("\n✅ MODEL SAVED SUCCESSFULLY")

# =========================================================
# LOAD MODEL
# =========================================================

loaded_model = joblib.load("tamilnadu_best_model.pkl")

print("\n✅ MODEL LOADED SUCCESSFULLY")

# =========================================================
# END OF PROJECT
# =========================================================
