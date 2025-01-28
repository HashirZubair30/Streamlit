import streamlit as st
import pickle
import numpy as np

# Load the trained SVM model
with open("/content/svm_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title of the Streamlit app
st.title("SVM Model Predictor")

# Input fields for user input
st.header("Enter the Input Features")
# Add appropriate input fields based on your SVM model's features
feature_1 = st.number_input("Enter Feature 1", min_value=0.0, max_value=100.0, step=0.1)
feature_2 = st.number_input("Enter Feature 2", min_value=0.0, max_value=100.0, step=0.1)
# Add more fields as needed for your SVM model inputs

# When user clicks the Predict button
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([[feature_1, feature_2]])
    prediction = model.predict(input_data)

    # Display the prediction result
    st.subheader("Prediction")
    st.write(f"The model predicts: {prediction[0]}")
