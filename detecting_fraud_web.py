import streamlit as st
import pandas as pd
import joblib as jl

model = jl.load("fraud_detection_pipeline.pkl")
st.title("Detecting Financial Fraud")
st.markdown("Financial Transactions Details HERE ")
st.divider()
transaction_type = st.selectbox("Type of Transaction",["PAYMENT","TRANSFER","CASHOUT","DEPOSITS"])
amount = st.number_input("amount",min_value = 0.0,value=1000.00)
oldbalanceOrg = st.number_input("OlD Blanace (SENDER)",min_value = 0.0,value=1000.00)
newbalanceOrig = st.number_input("NEW Balance (SENDER)",min_value = 0.0,value=9000.00)
oldbalanceDest = st.number_input("OLD Balance (RECEIVER)",min_value = 0.0,value=0.0)
newbalanceDest = st.number_input("NEW Balance (RECEIVER)",min_value = 0.0,value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type" : transaction_type,
        "amount" : amount,
        "newbalanceDest":newbalanceDest,
        "oldbalanceDest":oldbalanceDest,
        "oldbalanceOrg":oldbalanceOrg,
        "newbalanceOrig":newbalanceOrig
    }])
    prediction = model.predict(input_data)[0]
    st.subheader(f"Predction: 'FRAUD ! !   OR   CLEAN + +'")

    if prediction == 1:
        st.error("FRAUD !!!")
    else:
        st.success("CLEAN ++")
