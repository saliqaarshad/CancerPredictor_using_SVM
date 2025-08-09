import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st

# Load and prepare the dataset
df = pd.read_csv("KNNAlgorithmDataset.csv")
# Separate features and target
df=df.drop(columns=['Unnamed: 32'])
df.dropna()
X = df.drop(columns=['diagnosis', 'id'])
y = df['diagnosis'].apply(lambda x: 1 if x=='B' else 0).values
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
# Train the KNN model
model = SVC()
model.fit(X_train, y_train)
# Save the model
joblib.dump(model, 'MODEL.pkl')  # Corrected: changed from dumb to dump

