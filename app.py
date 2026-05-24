import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("eci_results_tamilnadu_2026.csv")

print(df.head())

print(df.info())

print(df.shape)

print(df.columns)
print(df.isnull().sum())
df = df.dropna()

# Create winner column
# Candidate with highest votes in each constituency is winner

max_votes = df.groupby('Constituency')['Total Votes'].transform('max')

df['Winner'] = np.where(df['Total Votes'] == max_votes, 1, 0)

print(df[['Constituency', 'Candidate', 'Total Votes', 'Winner']].head())

# Save original party names
party_names = df['Party']

# Encode party names
party_encoder = LabelEncoder()
df['Party_Encoded'] = party_encoder.fit_transform(df['Party'])

# Display party mapping
party_mapping = pd.DataFrame({
    'Party Name': party_encoder.classes_,
    'Encoded Value': range(len(party_encoder.classes_))
})

print(party_mapping)

const_encoder = LabelEncoder()
df['Constituency_Encoded'] = const_encoder.fit_transform(df['Constituency'])

#Select Features and Target
X = df[[
    'EVM Votes',
    'Postal Votes',
    'Total Votes',
    '% Votes',
    'Party_Encoded',
    'Constituency_Encoded'
]]

y = df['Winner']

#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Train Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print('Logistic Regression Accuracy:', accuracy_score(y_test, lr_pred))

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print('Random Forest Accuracy:', accuracy_score(y_test, rf_pred))

xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print('XGBoost Accuracy:', accuracy_score(y_test, xgb_pred))

lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

models = ['Logistic Regression', 'Random Forest', 'XGBoost']
accuracies = [lr_acc, rf_acc, xgb_acc]

plt.figure(figsize=(8,5))
plt.bar(models, accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

party_votes = df.groupby('Party')['Total Votes'].sum().sort_values(ascending=False).head(10)

party_votes.plot(kind='bar', figsize=(12,6))
plt.title('Top 10 Parties by Total Votes')
plt.xlabel('Party')
plt.ylabel('Votes')
plt.show()

cm = confusion_matrix(y_test, xgb_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(y_test, xgb_pred))

#Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})

importance = importance.sort_values(by='Importance', ascending=False)

print(importance)

#Predict Future Election Winner
sample_data = pd.DataFrame({
    'EVM Votes': [70000],
    'Postal Votes': [500],
    'Total Votes': [70500],
    '% Votes': [40.2],
    'Party_Encoded': [86],
    'Constituency_Encoded': [3]
})

prediction = xgb_model.predict(sample_data)

if prediction[0] == 1:
    print('Predicted Result: Winner')
else:
    print('Predicted Result: Not Winner')

    encoded_party = 86

actual_party_name = party_encoder.inverse_transform([encoded_party])

print('Party Name:', actual_party_name[0])

import joblib

joblib.dump(xgb_model, 'tamilnadu_election_model.pkl')

print('Model saved successfully')

loaded_model = joblib.load('tamilnadu_election_model.pkl')