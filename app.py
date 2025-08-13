import pickle
import streamlit as st
import pandas as pd

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Titanic Survival Prediction")

# Create input fields matching your features
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of siblings/spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of parents/children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert inputs to model format
sex_val = 1 if sex == "male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_val = embarked_map[embarked]

input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_val],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked_val]
})

if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("The passenger is predicted to survive!")
    else:
        st.error("The passenger is predicted not to survive.")