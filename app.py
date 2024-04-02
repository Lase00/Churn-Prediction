import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO

# Load the trained model
# URL of the raw pickle file in your GitHub repository
url = 'https://raw.githubusercontent.com/<username>/<repository>/<branch>/finalized_logistic_regression_model.pkl'


with open("finalized_logistic_regression_model.pkl", 'rb') as file:
    lr_model = pickle.load(file)

# Function to create sample customer
feature_names = ['Contract', 'MonthlyCharges', 'tenure', 'OnlineSecurity', 'PhoneService', 'TechSupport', 'PaperlessBilling', 'TotalCharges', 'OnlineBackup', 'InternetService', 'SeniorCitizen', 'DeviceProtection']


def create_sample_customer(InternetService, Contract,
                           OnlineSecurity, TechSupport, OnlineBackup,
                           DeviceProtection,PaperlessBilling, PhoneService, 
                           SeniorCitizen, MonthlyCharges, TotalCharges,
                           tenure):
    # Updated mappings to include 'No internet service' and remove 'PaymentMethod'
    mappings = {
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1}
    }
    
    # Convert categorical and binary features to numerical
    sample_data = [
        mappings['Contract'].get(Contract, -1),
        MonthlyCharges,
        tenure,
        mappings['OnlineSecurity'].get(OnlineSecurity, -1),
        mappings['PhoneService'].get(PhoneService, -1),
        mappings['TechSupport'].get(TechSupport, -1),
        mappings['PaperlessBilling'].get(PaperlessBilling, -1),
        TotalCharges,
        mappings['OnlineBackup'].get(OnlineBackup, -1),
        mappings['InternetService'].get(InternetService, -1),
        SeniorCitizen,
        mappings['DeviceProtection'].get(DeviceProtection, -1),
    ]
    return sample_data

# Sample usage:
sample_customer = create_sample_customer(
    InternetService='Fiber optic', 
    Contract='Month-to-month',
    OnlineSecurity='No',
    TechSupport='Yes', 
    OnlineBackup='Yes', 
    DeviceProtection='Yes', 
    PaperlessBilling='Yes',
    PhoneService='No',
    SeniorCitizen=0,
    MonthlyCharges=70, 
    TotalCharges=700, 
    tenure=10
)
# Convert the sample data to a DataFrame with correct column names
sample_customer_df = pd.DataFrame([sample_customer], columns=feature_names)

# Function to generate recommendations
churn_prediction = lr_model.predict(sample_customer_df)

# Interpreting the prediction
prediction_text = "will churn" if churn_prediction[0] == 1 else "will not churn"
print(f"The customer {prediction_text}.")

# Function to generate recommendations
def generate_recommendations(customer_features):
    recommendations = []
    # Example conditions for recommendations
    if customer_features[9] == 0:  # Assuming 1 is 'Fiber optic'
        recommendations.append("Consider offering a promotional deal on fiber optic services.")
    if customer_features[0] == 0:  # Assuming 0 is 'Month-to-month'
        recommendations.append("Recommend switching to a longer-term contract for better stability.")
    if customer_features[8] == 0:  # Assuming this is 'OnlineBackup'
        recommendations.append("Offer a discount on Online Backup services to enhance value.")
    if customer_features[11] == 0:  # No Device Protection
        recommendations.append("Suggest adding device protection to their plan for peace of mind.")
    if customer_features[5] == 0:  # No Tech Support
        recommendations.append("Advise the importance of tech support for uninterrupted service.")
    if customer_features[2] <= 12:  # Less than 1 year
        recommendations.append("Extend a loyalty program invitation after the first year.")
    if customer_features[1] > 75:  # High Monthly Charges
        recommendations.append("Review the customer's plan to offer a competitive rate.")    
    
    return recommendations if recommendations else ["Customer Was Retained Amazing!!!!!"]

# Generate recommendations if the customer is predicted to churn
if churn_prediction[0] == 1:
    recommendations = generate_recommendations(sample_customer)
    print("Recommendations to prevent churn:")
    for recommendation in recommendations:
        print(f"- {recommendation}")

# Streamlit UI
st.title("Customer Churn Prediction and Recommendations")

# Collect user inputs for each feature
InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
OnlineSecurity = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
PhoneService = st.selectbox('Phone Service', ['Yes', 'No'])
SeniorCitizen = st.radio('Senior Citizen', [1, 0])
MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0)
tenure = st.number_input('Tenure (in months)', min_value=0)
# Calculate total charges and display it
total_charges = MonthlyCharges * tenure
st.write(f"Calculated Total Charges: {total_charges}")

# Button to make prediction
if st.button('Predict Churn and Generate Recommendations'):
    sample_customer = create_sample_customer(
        InternetService, 
        Contract,
        OnlineSecurity, 
        TechSupport, 
        OnlineBackup,
        DeviceProtection, 
        PaperlessBilling,
        PhoneService,
        SeniorCitizen,
        MonthlyCharges, 
        total_charges, 
        tenure
    )
    
    churn_prediction = lr_model.predict([sample_customer])
    prediction_text = "will churn" if churn_prediction[0] == 1 else "will not churn"
    st.write(f"The customer {prediction_text}.")

    # Generate and display recommendations if the customer will churn
    if churn_prediction[0] == 1:
        recommendations = generate_recommendations(sample_customer)
        st.write("Recommendations to prevent churn:")
        for recommendation in recommendations:
            st.write(f"- {recommendation}")
