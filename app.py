import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


st.title("Titanic Survival Prediction App")
st.write("Enter passenger details to see if they would have survived the Titanic disaster.")

# User input form
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode input
sex_encoded = 1 if sex == "Male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_map[embarked]

# Predict
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("The passenger **would have survived**.")
    else:
        st.error("The passenger **would not have survived**.")
