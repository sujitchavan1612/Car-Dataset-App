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
page = st.sidebar.selectbox("Select Page", ["Data Preview", "Visualizations", "Modeling", "Prediction"])
uploaded_file = st.sidebar.file_uploader("Upload your car dataset CSV", type=["csv"])

# Shared logic
model = None
scaler = None
X = None
df = None
df_encoded = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()

    # Create binary target
    df["Target"] = (df["Price"] > 15000).astype(int)

    # Drop potential data leakage columns before encoding
    df = df.drop(["Price", "Model"], axis=1, errors="ignore")

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("Target", axis=1)
    y = df_encoded["Target"]

    # Inject random noise for realism testing
    for i in range(2):
        np.random.seed(i)
        X[f"noise_{i}"] = np.random.rand(X.shape[0])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model training
    model = LogisticRegression(max_iter=100, C=0.5, solver="liblinear")
    model.fit(X_train, y_train)

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
                sns.scatterplot(data=df, x="EngineV", y="Target", hue="Target", ax=ax1)
                st.pyplot(fig1)

        with col2:
            if "Mileage" in df.columns:
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
    if uploaded_file is not None and model is not None:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.metric(label="🎯 Model Accuracy", value=f"{acc:.2f}")

        if acc > 0.90:
            st.warning("⚠️ Accuracy above 90%. Consider reducing strong predictors.")
        elif acc < 0.80:
            st.warning("⚠️ Accuracy below 80%. Add useful features or reduce noise.")
        else:
            st.success("✅ Accuracy is between 80% and 90% — acceptable for real-world data.")

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("🔍 Confusion Matrix")
        fig4, ax4 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax4)
        st.pyplot(fig4)

        st.subheader("📊 Feature Importance")
        coef = pd.Series(model.coef_[0], index=X.columns)
        st.bar_chart(coef.sort_values(ascending=False))
    else:
        st.info("Upload dataset to train the model.")

# Page 4 - Prediction
elif page == "Prediction":
    st.title("🔮 Predict Car Price Category")
    if uploaded_file is not None and model is not None and scaler is not None:
        with st.form("prediction_form"):
            brand = st.selectbox("Brand", df["Brand"].unique())
            body = st.selectbox("Body Type", df["Body"].unique())
            mileage = st.number_input("Mileage", min_value=0)
            engine_type = st.selectbox("Engine Type", df["Engine Type"].unique())
            registration = st.selectbox("Registered?", df["Registration"].unique())
            year = st.number_input("Year", min_value=1980, max_value=2025)

            submit = st.form_submit_button("Predict")

        if submit:
            # Create input dict
            input_dict = {
                "Mileage": mileage,
                "Year": year,
                f"Brand_{brand}": 1,
                f"Body_{body}": 1,
                f"Engine Type_{engine_type}": 1,
                f"Registration_{registration}": 1
            }

            # Remove unseen keys
            input_dict = {k: v for k, v in input_dict.items() if k in X.columns}
            input_df = pd.DataFrame([input_dict])

            for col in X.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[X.columns]  # ensure column order

            # Scale and predict
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]

            st.subheader("🧾 Prediction Result")
            st.success(f"Predicted Price Category: {'Above 15,000' if pred==1 else '15,000 or Below'}")

            if prob > 0.85:
                st.success(f"Confidence: {prob:.2%} — Very Strong Prediction")
            elif prob < 0.6:
                st.warning(f"Confidence: {prob:.2%} — Weak Prediction, revise input or model")
            else:
                st.info(f"Confidence: {prob:.2%}")
    else:
        st.info("Upload a dataset and train the model to make predictions.")
