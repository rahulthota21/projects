import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset function
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/rahulthota21/ProjectVault/main/Cricket%20Match%20Outcome%20Predictor/Match_Prediction.csv'
    data = pd.read_csv(url)
    return data

# Streamlit App
st.set_page_config(page_title="Cricket Match Prediction", layout="wide")
st.title("🏏 Cricket Match Prediction App")
st.markdown("""
This app predicts match outcomes using Machine Learning models. 
It also provides insightful visualizations to understand the data.
""")

# Load and display the dataset
data = load_data()
st.sidebar.header("🗂️ Dataset Overview")
st.sidebar.write("Explore the dataset used for predictions.")

if st.sidebar.checkbox("Show Dataset"):
    st.write("### Dataset Sample")
    st.write(data.head())

# Visualizations
st.sidebar.header("📊 Data Visualizations")
st.sidebar.write("Select visualization options:")

if st.sidebar.checkbox("Correlation Heatmap"):
    st.write("### Correlation Heatmap")
    data_encoded = data.copy()
    categorical_columns = data.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        for col in categorical_columns:
            data_encoded[col] = LabelEncoder().fit_transform(data[col])
    numeric_data = data_encoded.select_dtypes(include=['float64', 'int64'])
    if numeric_data.empty:
        st.warning("No numerical columns available for correlation heatmap.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

if st.sidebar.checkbox("Boxplot of Numeric Features"):
    st.write("### Boxplot of Numeric Features")
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    selected_column = st.sidebar.selectbox("Select Numeric Column for Boxplot", numeric_columns)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=data, x=selected_column, ax=ax)
    ax.set_title(f"Boxplot of {selected_column}")
    st.pyplot(fig)

if st.sidebar.checkbox("Countplot of Categorical Features"):
    st.write("### Countplot of Categorical Features")
    categorical_columns = data.select_dtypes(include=['object']).columns
    selected_column = st.sidebar.selectbox("Select Categorical Column for Countplot", categorical_columns)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=data, x=selected_column, ax=ax)
    ax.set_title(f"Countplot of {selected_column}")
    ax.bar_label(ax.containers[0])
    st.pyplot(fig)

if st.sidebar.checkbox("Scatterplot"):
    st.write("### Scatterplot")
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    col_x = st.sidebar.selectbox("Select X-Axis", numeric_columns)
    col_y = st.sidebar.selectbox("Select Y-Axis", numeric_columns)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x=col_x, y=col_y, ax=ax)
    ax.set_title(f"Scatterplot: {col_x} vs {col_y}")
    st.pyplot(fig)

# Prediction Inputs
st.sidebar.header("🔮 Prediction Inputs")
st.sidebar.write("Provide inputs to predict match outcomes.")

inputs = {}
match_format = st.sidebar.selectbox("Select Match Format", {"Test": 0, "ODI": 1, "T20": 2})
inputs["Match_Format"] = match_format

pitch_type = st.sidebar.selectbox("Select Pitch Type", {0: "Green", 1: "Dry", 2: "Flat"})
inputs["Pitch_Type"] = pitch_type

recent_form = st.sidebar.slider("Select Recent Form (0-5)", 0, 5, 3)
inputs["Recent_Form"] = recent_form

st.sidebar.subheader("📌 Additional Features")
for col in data.columns:
    if col not in ["Match_Outcome", "Match_Format", "Pitch_Type", "Recent_Form"]:
        if data[col].dtype == 'object':
            inputs[col] = st.sidebar.selectbox(f"{col}", data[col].unique())
        else:
            inputs[col] = st.sidebar.number_input(
                f"{col}", float(data[col].min()), float(data[col].max()), float(data[col].mean())
            )

# Model Selection
st.sidebar.header("🧠 Model Selection")
model_name = st.sidebar.selectbox("Choose a Machine Learning Model", ["Logistic Regression", "Random Forest"])

# Preprocessing function
def preprocess_inputs(inputs):
    match_format_map = {"Test": 0, "ODI": 1, "T20": 2}
    inputs["Match_Format"] = match_format_map.get(inputs["Match_Format"], -1)
    if inputs["Match_Format"] == -1:
        raise ValueError("Invalid Match_Format. Please select 'Test', 'ODI', or 'T20'.")
    return inputs

# Training and prediction
def train_and_predict(model_name, data, inputs):
    inputs = preprocess_inputs(inputs)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    categorical_columns = X.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        for col in categorical_columns:
            X[col] = LabelEncoder().fit_transform(X[col])
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    user_input = pd.DataFrame([inputs])
    for col in categorical_columns:
        if col in user_input.columns:
            user_input[col] = LabelEncoder().fit(data[col]).transform(user_input[col])

    for col in X.columns:
        if col not in user_input.columns:
            user_input[col] = 0

    user_input = user_input[X.columns]
    user_input_scaled = scaler.transform(user_input)

    model = LogisticRegression() if model_name == "Logistic Regression" else RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    prediction = model.predict(user_input_scaled)

    return y_test, y_pred, prediction

if st.sidebar.button("🚀 Predict Outcome"):
    try:
        y_test, y_pred, prediction = train_and_predict(model_name, data, inputs)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model Accuracy: {accuracy:.2%}")

        report = classification_report(y_test, y_pred, output_dict=True)
        st.write("#### Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

        outcome_map = {1: "Team A Wins", 0: "Team B Wins", 2: "Draw"}
        st.markdown(f"### 🎉 Predicted Outcome: **{outcome_map.get(prediction[0], 'Unknown')}**")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Option to download results
if st.sidebar.button("📥 Download Results"):
    st.write("Feature coming soon!")
