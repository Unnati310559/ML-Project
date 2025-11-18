import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Diabetes ML Project", layout="wide")

st.title("Diabetes Classification Web App")
st.write("Built with Streamlit + Scikit-Learn")

# -----------------------------



# Upload Dataset
# -----------------------------
uploaded = st.file_uploader("Upload the diabetes dataset (.csv)", type=["csv"])

if uploaded:
    data = pd.read_csv(uploaded)
    st.subheader("📌 Dataset Preview")
    st.dataframe(data.head())

    # -----------------------------
    # Preprocessing
    # -----------------------------
    st.subheader("🧹 Data Preprocessing")

    zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for col in zero_cols:
        data[col] = data[col].replace(0, data[col].median())

    st.write("Zero values in columns replaced with median.")

    # -----------------------------
    # Visualization
    # -----------------------------
    st.subheader("📊 Exploratory Data Analysis")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

    st.subheader("📚 Model Training & Evaluation")

    model_list = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Support Vector Classifier (SVC)": SVC(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier()
    }

    results = {}

    for model_name, model in model_list.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        results[model_name] = [acc, prec, rec]

    results_df = pd.DataFrame(
        results,
        index=["Accuracy", "Precision", "Recall"]
    ).T

    st.write("### 📄 Model Comparison Table")
    st.dataframe(results_df)

    # -----------------------------
    # Bar chart for model comparison
    # -----------------------------
    st.subheader("📈 Accuracy Comparison")

    fig2, ax2 = plt.subplots(figsize=(7,5))
    ax2.bar(results_df.index, results_df["Accuracy"])
    plt.xticks(rotation=15)
    plt.ylabel("Accuracy")
    st.pyplot(fig2)

    # -----------------------------
    # Prediction Section
    # -----------------------------
    st.subheader("🔮 Make a Prediction")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20)
        Glucose = st.number_input("Glucose", 0, 300)
        BloodPressure = st.number_input("Blood Pressure", 0, 200)
        SkinThickness = st.number_input("Skin Thickness", 0, 100)

    with col2:
        Insulin = st.number_input("Insulin", 0, 900)
        BMI = st.number_input("BMI", 0.0, 70.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 5.0)
        Age = st.number_input("Age", 1, 120)

    # Convert to array
    input_data = np.array([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Use best model
    best_model_name = results_df["Accuracy"].idxmax()
    best_model = model_list[best_model_name]

    if st.button("Predict"):
        pred = best_model.predict(input_scaled)
        if pred[0] == 1:
            st.error(f"Result: **Diabetes Detected** 🚨 (Model: {best_model_name})")
        else:
            st.success(f"Result: **No Diabetes** 😊 (Model: {best_model_name})")

else:
    st.info("Please upload a dataset to continue.")
