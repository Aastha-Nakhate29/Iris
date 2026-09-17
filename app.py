import streamlit as st
import numpy as np
import joblib

#load model
model =joblib.load("iris_model.pkl")

#page title
st.title("Machine Learning on Iris Dataset")

#input labels
sepal_length = st.number_input("Sepal Length (cm)")
sepal_width = st.number_input("Sepal Width (cm)")
petal_length = st.number_input("Petal Length (cm)")
petal_width = st.number_input("Petal Width (cm)")

#prediction
if st.button("Predict"):    
    input_data = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction = model.predict(input_data)
    st.write(f"The predicted species is: {prediction[0]}")
