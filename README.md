# 79592025_Churning_Customers
# Customer Churn Prediction App

## Project Description

The Customer Churn Prediction App is a web-based application that predicts the likelihood of a customer churning given some customer profile details. This application employs a machine learning model based on the Keras library to provide valuable insights for customer retention strategies.

## Functionality

1. **User Input:**
   - Users input customer details through a user-friendly interface. Key features include Online Security, Contract type, Payment Method, Tenure, Monthly Charges, and Total Charges.

2. **Data Processing:**
   - Categorical values are converted to numerical representations for model compatibility.
   - Total Charges are calculated based on Tenure and Monthly Charges.

3. **Prediction:**
   - The app triggers a prediction when the user clicks the "Predict Customer Churn" button.
   - The model, loaded from `model.h5`, processes the input and returns a churn probability.

4. **Confidence Factor:**
   - The app calculates a confidence factor, indicating the model's certainty in its prediction.

5. **Outcome Display:**
   - The app displays the churn probability and confidence factor.
   - It provides messages indicating the likelihood of churn and a confidence percentage.

## Usage

1. **Installation:**
   ```bash
   pip install streamlit pandas numpy scikit-learn tensorflow
   ```

2. **Run the App:**
   ```bash
   streamlit run app.py
   ```

3. **Interact with the App:**
   - Open the provided link in your browser and complete the customer details.
   - Click the "Predict Customer Churn" button to see the prediction and confidence factor.

## Video Demonstration

[Watch the Video Demonstration](https://youtu.be/fPnywL4aGkI)

## Dependencies

- Streamlit
- Pandas
- NumPy
- scikit-learn
- TensorFlow

