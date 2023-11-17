# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the trained model
model = load_model('model.h5')

# Load the scaler using pickle
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


def main():
    st.title("Customer Churn App")
    st.markdown("""
        :dart: This app predicts customer churn in an organization
    """)
    st.write("Complete the customer details from the menu below.")

    # Create input fields for user features

    # Convert "OnlineSecurity" categorical value to numerical
    OnlineSecurity_mapping = {'no': 0, 'yes': 1, 'no internet service': 2}
    OnlineSecurity = st.selectbox(
        'Choose customer online security:', list(OnlineSecurity_mapping.keys()))
    OnlineSecurity = OnlineSecurity_mapping[OnlineSecurity]

    # Convert "Contract" categorical value to numerical
    contract_mapping = {'month-to-month': 0, 'one_year': 1, 'two_years': 2}
    Contract = st.selectbox('Choose customer contract:',
                            list(contract_mapping.keys()))
    Contract = contract_mapping[Contract]

    # Convert "PaymentMethod" categorical value to numerical
    PaymentMethod_mapping = {
        'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3, 'Electronic check': 0, 'Mailed check': 1}
    PaymentMethod = st.selectbox(
        'Choose customer payment method:', list(PaymentMethod_mapping.keys()))
    PaymentMethod = PaymentMethod_mapping[PaymentMethod]

    Tenure = st.number_input(
        'Number of months the customer has been with the organization:', min_value=0, max_value=100, value=0)
    MonthlyCharges = st.number_input(
        'Monthly charges:', min_value=0, max_value=250000, value=0)

    TotalCharges = Tenure * MonthlyCharges

    # Display TotalCharges
    st.success(f'TotalCharges: {TotalCharges}')

    # Make prediction when the user clicks the button
    if st.button('Predict Customer Churn'):

        values = [int(OnlineSecurity), int(Contract), int(
            PaymentMethod), float(MonthlyCharges), float(TotalCharges), int(Tenure)]

        print(values)
        values = scaler.transform([values])[0]
        print(values)
        prediction = model.predict(np.array([values]))

        result = prediction[0][0]
        print(result)

        # Display a message based on the churn probability (result)
        confidence = (1 - result) * 100 if result < 0.5 else result * 100
        print(confidence)

        prediction = int(result > 0.5)

        st.success(f"Churn Probability: {np.round(prediction)}")

        st.warning(f"Confidence Factor: {round(confidence, 2)}%")

        if (prediction == 1):
            st.success(
                f"There is a {round(confidence, 2)}% chance that customer will Churn")
        else:
            st.success(
                f"There is a {round(confidence, 2)}% chance that this customer will not Churn")


if __name__ == "__main__":
    main()
