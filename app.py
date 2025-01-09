import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# Load the trained model
# URL of the raw pickle file in your GitHub repository
url = 'https://raw.githubusercontent.com/Lase00/Churn-Prediction/main/finalized_logistic_regression_model.pkl'

# Fetch the pickle file
response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful

# Load the model
lr_model = joblib.load(BytesIO(response.content))

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

# Function to generate insights
churn_prediction = lr_model.predict(sample_customer_df)

# Interpreting the prediction
prediction_text = "will churn" if churn_prediction[0] == 1 else "will not churn"
print(f"The customer {prediction_text}.")

# Function to generate insights
def generate_insights(customer_features):
    insights = []
    # Example conditions for insights
    if customer_features[9] == 0:  # Assuming 1 is 'Fiber optic'
        insights.append("The customer may value cost-effective options over high-speed internet like fiber optic.")
    if customer_features[0] == 0:  # Assuming 0 is 'Month-to-month'
        insights.append("The customer prefers flexibility with month-to-month contracts rather than long-term commitments.")
    if customer_features[8] == 0:  # Assuming this is 'OnlineBackup'
        insights.append("The customer might not prioritize data security solutions like Online Backup services.")
    if customer_features[11] == 0:  # No Device Protection
        insights.append("The customer may not see device protection as a necessary value addition to their plan.")
    if customer_features[5] == 0:  # No Tech Support
        insights.append("The customer may rely on self-resolution methods instead of Tech Support services.")
    if customer_features[2] <= 12:  # Less than 1 year
        insights.append("The customer is relatively new and may still be exploring the service's value proposition.")
    if customer_features[1] > 75:  # High Monthly Charges
        insights.append("The customer incurs high monthly charges, which might impact their satisfaction level.")    
    
    return insights if insights else ["Customer Was Retained Amazing!!!!!"]

# Generate insights if the customer is predicted to churn
if churn_prediction[0] == 1:
    insights = generate_insights(sample_customer)
    print("Insights From Churned Customer:")
    for insights in insights:
        print(f"- {insights}")

# Streamlit UI
st.title("Customer Churn Prediction with Actionable Insights for Data-Driven Decisions")

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
if st.button('Predict Churn and Generate insights'):
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

    # Generate and display insights if the customer will churn
    if churn_prediction[0] == 1:
        insights = generate_insights(sample_customer)
        st.write("Actionable Insights to Prevent Customer Churn:")
        for insights in insights:
            st.write(f"- {insights}")
