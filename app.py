import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.title("Car Dataset - Logistic Regression Model")

# Upload CSV
uploaded_file = st.file_uploader("Upload your car dataset CSV", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Drop missing
    df = df.dropna()

    # Create binary target
    df["Target"] = (df["Price"] > 15000).astype(int)

    # Drop high-cardinality or ID-like features
    if "Model" in df.columns:
        df = df.drop("Model", axis=1)

    # Encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    # Split features and target
    X = df.drop("Target", axis=1)
    y = df["Target"]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    st.subheader("Model Evaluation")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Visualization
    st.subheader("Target Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Target", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pd.DataFrame(X).corr(), annot=True, fmt=".2f", ax=ax2)
    st.pyplot(fig2)
else:
    st.info("Upload a CSV file to proceed.")
