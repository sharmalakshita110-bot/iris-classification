import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Classification App")

st.write("Enter flower measurements below:")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)")
sepal_width = st.number_input("Sepal Width (cm)")
petal_length = st.number_input("Petal Length (cm)")
petal_width = st.number_input("Petal Width (cm)")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    classes = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Flower Type: {classes[prediction[0]]}")
