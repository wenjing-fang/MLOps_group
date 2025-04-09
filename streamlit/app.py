##add the path to the folder mlops project to identify the package or is there other method to adress this issue?
import sys
import os
sys.path.append('/Users/fionafang/Documents/A_OnUsing/Study_ESSEC/Course-8_MLOps/MLOps_group')
print("Current working directory:", os.getcwd())
print("Current sys.path:", sys.path)
###################


import project_code.functions as f
import project_code.models as models 

import streamlit as st
import pandas as pd
import os 

# Load the model according to user's preference
filenames=os.listdir('./checkpoints')
modelnames=[f[:f.index('.')] for f in filenames]
modelname= st.selectbox("Choose our existing model for prediction", modelnames)
model=f.load_model(modelname)

st.title("🏦 Predict Bank Marketing Response")
st.header("Please fill out the client information:")

#Numeric data
age = st.number_input("Age", min_value=18, max_value=150, value=30)
balance = st.number_input("Account Balance", value=100)
day = st.number_input("Contact Day", min_value=1, max_value=31, value=15)
duration = st.number_input("Last Contact Duration (sec)", value=180)
campaign = st.number_input("Number of Contacts During Campaign", value=1)
pdays = st.number_input("Days Since Last Contact", value=-1)
previous = st.number_input("Number of Previous Contacts", value=0)


# Categorical data
job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
    "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("Has Credit in Default?", ["yes", "no"])
housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
loan = st.selectbox("Has Personal Loan?", ["yes", "no"])
contact = st.selectbox("Contact Communication Type", ["cellular", "telephone","unknown"])
month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun","jul", "aug", "sep", "oct", "nov", "dec"])
poutcome = st.selectbox("Previous Outcome", ["unknown", "other", "failure", "success"])

## predict based on input data
if st.button("Predict"):
    # Combine features into a single row
    input_data = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
        'balance': balance, 'housing': housing, 'loan': loan, 'contact': contact,
        'day': day, 'month': month, 'duration': duration, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }])
    processed_data=f.preprocess_user_data(input_data)


    # Use model pipeline to predict
    prediction = model.predict(processed_data)
    print(prediction)
    result= 'yes' if prediction==1 else 'no'
    st.success(f"Is this customer likely to subscribe a term deposit: {result}")
