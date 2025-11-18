import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("🤖 Diabetes Prediction Machine Learning App")
st.write("Dataset loads automatically → Models train → Predict")

# ----------------------------
# Load Dataset (Automatically)
# ----------------------------
DATA_PATH = "diabetes.csv"   # keep file in same folder

try:
    data = pd.read_csv(DATA_PATH)
except:
    st.error("❌ Dataset not found! Please make sure 'diabetes.csv' is in the same folder.")
    st.stop()

st.subheader("📌 Dataset Preview")
st.dataframe(data.head())

# ----------------------------
# Data Preprocessing
# ----------------------------
st.subheader("🧹 Data Preprocessing")

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    data[col] = data[col].replace(0, data[col].median())

st.success("Zero values replaced with medians.")

# ----------------------------
# EDA
# ----------------------------
st.subheader("📊 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# ----------------------------
# Split Data
# ----------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# ----------------------------
# Train Models
# ----------------------------
st.subheader("📚 Model Training & Evaluation")

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVC": SVC(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    results[name] = [acc, prec, rec]

results_df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall"]).T
st.dataframe(results_df)

# ----------------------------
# Prediction
# ----------------------------
st.subheader("🔮 Make a Prediction")

col1, col2 = st.columns(2)

with col1:
    Preg = st.number_input("Pregnancies", 0, 20)
    Glu = st.number_input("Glucose", 0, 300)
    BP = st.number_input("Blood Pressure", 0, 200)
    Skin = st.number_input("Skin Thickness", 0, 100)

with col2:
    Insulin = st.number_input("Insulin", 0, 900)
    BMI = st.number_input("BMI", 0.0, 70.0)
    DPF = st.number_input("Diabetes Pedigree Function", 0.0, 5.0)
    Age = st.number_input("Age", 1, 120)

user_input = np.array([[Preg, Glu, BP, Skin, Insulin, BMI, DPF, Age]])
user_scaled = scaler.transform(user_input)

best_model_name = results_df["Accuracy"].idxmax()
best_model = models[best_model_name]

if st.button("Predict"):
    pred = best_model.predict(user_scaled)
    if pred[0] == 1:
        st.error(f"🚨 **Diabetes Detected** (Model: {best_model_name})")
    else:
        st.success(f"✅ **No Diabetes** (Model: {best_model_name})")
