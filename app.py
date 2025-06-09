import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Car Classifier", layout="wide")

# Sidebar
page = st.sidebar.selectbox("Select Page", ["Data Preview", "Visualizations", "Modeling"])
uploaded_file = st.sidebar.file_uploader("Upload your car dataset CSV", type=["csv"])

# Shared logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()
    df["Target"] = (df["Price"] > 15000).astype(int)

    if "Model" in df.columns:
        df = df.drop("Model", axis=1)

    # Encode categoricals
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("Target", axis=1)
    y = df_encoded["Target"]

    # Restore mildly predictive features
    for col in ["Mileage", "Year"]:
        if col not in X.columns and col in df_encoded.columns:
            X[col] = df_encoded[col]

    # Drop very strong predictors
    for col in ["Price", "EngineV"]:
        if col in X.columns:
            X = X.drop(col, axis=1)

    # Reduce noise (fewer noise columns)
    for i in range(2):
        np.random.seed(i)
        X[f"noise_{i}"] = np.random.rand(X.shape[0])

    # Train-test split (now with shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Page 1 - Data Preview
if page == "Data Preview":
    st.title("📊 Dataset Preview")
    if uploaded_file is not None:
        st.dataframe(df.head())
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.bar_chart(df["Target"].value_counts())
    else:
        st.info("Upload a CSV to begin.")

# Page 2 - Visualizations
elif page == "Visualizations":
    st.title("📈 Visualizations")
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            if "EngineV" in df.columns:
                st.write("Engine Size vs Price")
                fig1, ax1 = plt.subplots()
                sns.scatterplot(data=df, x="EngineV", y="Price", hue="Target", ax=ax1)
                st.pyplot(fig1)

        with col2:
            st.write("Mileage Distribution")
            fig2, ax2 = plt.subplots()
            sns.histplot(df["Mileage"], kde=True, ax=ax2)
            st.pyplot(fig2)

        st.subheader("Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)
    else:
        st.info("Upload a file to see plots.")

# Page 3 - Modeling
elif page == "Modeling":
    st.title("🤖 Logistic Regression Model")
    if uploaded_file is not None:
        # Slightly stronger model (higher C)
        model = LogisticRegression(max_iter=100, C=0.5, solver="liblinear")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric(label="🎯 Model Accuracy", value=f"{acc:.2f}")

        if acc > 0.90:
            st.warning("⚠️ Accuracy above 90%. Consider reducing features.")
        elif acc < 0.80:
            st.warning("⚠️ Accuracy still below 80%. Add mildly useful features or reduce noise.")
        else:
            st.success("✅ Accuracy is between 80% and 90% — perfect for real-world conditions.")

        # Classification Report
        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        fig4, ax4 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax4)
        st.pyplot(fig4)

        # Feature Importance
        st.subheader("📊 Feature Importance")
        coef = pd.Series(model.coef_[0], index=X.columns)
        st.bar_chart(coef.sort_values(ascending=False))
    else:
        st.info("Upload dataset to train the model.")
