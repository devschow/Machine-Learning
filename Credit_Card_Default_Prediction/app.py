import streamlit as st
import pandas as pd
import lightgbm as lgb 

import joblib
from sklearn.preprocessing import LabelEncoder

# Load the Random Forest model
@st.cache_data
def load_model():
    model = lgb
    model = joblib.load(open('lgb (1).pkl', 'rb'))
    return model

# Function to preprocess user input
def preprocess_input(data):
    # Encode categorical variables
    encoder = LabelEncoder()
    data['SEX'] = encoder.fit_transform(data['SEX'])
    data['EDUCATION'] = encoder.fit_transform(data['EDUCATION'])
    data['MARRIAGE'] = encoder.fit_transform(data['MARRIAGE'])
    
    return data

# Function to predict default
def predict_default(model, input_data):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]
    return prediction, probability

def main():
    st.markdown("<h1 style='text-align: center;'>Credit Card Default Prediction</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for user input
    #st.sidebar.title("Input Parameters")
    st.sidebar.markdown("<h1 style='text-align: center;'>Input Parameters</h1>", unsafe_allow_html=True)
    
    # Demographics section
    
    st.sidebar.subheader("Demographics")
    limit_bal = st.sidebar.number_input("Credit Limit", min_value=0, step=1000)
    gender_options = {1: "Male", 2: "Female"}
    sex = st.sidebar.radio("Gender", options=list(gender_options.values()), format_func=lambda x: gender_options[list(gender_options.keys())[list(gender_options.values()).index(x)]])
    education_options = {1: "Graduate school", 2: "University", 3: "High school", 4: "Others"}
    education = st.sidebar.radio("Education", options=list(education_options.values()), format_func=lambda x: education_options[list(education_options.keys())[list(education_options.values()).index(x)]])
    marriage_options = {1: "Married", 2: "Single", 3: "Others"}
    marriage = st.sidebar.radio("Marital Status", options=list(marriage_options.values()), format_func=lambda x: marriage_options[list(marriage_options.keys())[list(marriage_options.values()).index(x)]])
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)

    with st.container():
        col1, col2 = st.columns(2)    

    with col1:
        # Payment data section
        #st.header("Payment Data")
        st.subheader("Repayment status")
        repayment_legend = {-1: "Pay duly", 1: "Payment delay for one month", 2: "Payment delay for two months", 3: "Payment delay for three months", 4: "Payment delay for four months", 5: "Payment delay for five months", 6: "Payment delay for six months", 7: "Payment delay for seven months", 8: "Payment delay for eight months", 9: "Payment delay for nine months and above"}
        pay_0 = st.selectbox("September PAY", list(repayment_legend.keys()), format_func=lambda x: repayment_legend[x])
        pay_2 = st.selectbox("August PAY", list(repayment_legend.keys()), format_func=lambda x: repayment_legend[x])
        pay_3 = st.selectbox("July PAY", list(repayment_legend.keys()), format_func=lambda x: repayment_legend[x])
        pay_4 = st.selectbox("June PAY", list(repayment_legend.keys()), format_func=lambda x: repayment_legend[x])
        pay_5 = st.selectbox("May PAY", list(repayment_legend.keys()), format_func=lambda x: repayment_legend[x])
        pay_6 = st.selectbox("April PAY", list(repayment_legend.keys()), format_func=lambda x: repayment_legend[x])

    with col2:
        st.subheader("Bill Amount ($)")
        bill_amt1 = st.number_input("September BILL_AMT", min_value=0, step=1000)
        bill_amt2 = st.number_input("August BILL_AMT", min_value=0, step=1000)
        bill_amt3 = st.number_input("July BILL_AMT", min_value=0, step=1000)
        bill_amt4 = st.number_input("June BILL_AMT", min_value=0, step=1000)
        bill_amt5 = st.number_input("May BILL_AMT", min_value=0, step=1000)
        bill_amt6 = st.number_input("April BILL_AMT", min_value=0, step=1000)
    
    st.subheader("Amount of previous payments ($)")
    with st.container():
        col1,col2 = st.columns(2)
    with col1:

        pay_amt1 = st.number_input("September PAY_AMT", min_value=0, step=1000)
        pay_amt2 = st.number_input("August PAY_AMT", min_value=0, step=1000)
        pay_amt3 = st.number_input("July PAY_AMT", min_value=0, step=1000)
        
    with col2:
        pay_amt4 = st.number_input("June PAY_AMT", min_value=0, step=1000)
        pay_amt5 = st.number_input("May PAY_AMT", min_value=0, step=1000)
        pay_amt6 = st.number_input("April PAY_AMT", min_value=0, step=1000)
    
    # Horizontal line to separate sections
    st.markdown('<hr style="margin-top: 1rem; margin-bottom: 1rem; border: 1px solid #ccc;">', unsafe_allow_html=True)

    # Calculate Credit Utilization Ratio
    if limit_bal != 0:
        total_bill_amt = bill_amt1 + bill_amt2 + bill_amt3 + bill_amt4 + bill_amt5 + bill_amt6
        credit_utilization_ratio = total_bill_amt / limit_bal
    else:
        credit_utilization_ratio = 0
    
    # Create a dataframe with user input
    input_data = {
        'LIMIT_BAL': [limit_bal], 'SEX': [sex], 'EDUCATION': [education], 'MARRIAGE': [marriage],
        'AGE': [age], 'PAY_0': [pay_0], 'PAY_2': [pay_2], 'PAY_3': [pay_3], 'PAY_4': [pay_4], 'PAY_5': [pay_5], 'PAY_6': [pay_6],
        'BILL_AMT1': [bill_amt1], 'BILL_AMT2': [bill_amt2], 'BILL_AMT3': [bill_amt3],
        'BILL_AMT4': [bill_amt4], 'BILL_AMT5': [bill_amt5], 'BILL_AMT6': [bill_amt6],
        'PAY_AMT1': [pay_amt1], 'PAY_AMT2': [pay_amt2], 'PAY_AMT3': [pay_amt3],
        'PAY_AMT4': [pay_amt4], 'PAY_AMT5': [pay_amt5], 'PAY_AMT6': [pay_amt6],
        'Credit_Utilization_Ratio': [credit_utilization_ratio]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Preprocess the input data
    input_df = preprocess_input(input_df)
    
    # Load the model
    model = load_model()
    
    # Initialize probability to 0%
    probability = [0.0]
    
    # Predict
    if input_df.shape[0] != 0:
        prediction, probability = predict_default(model, input_df)
    
    # Prediction Result
    st.header("Prediction Result")
    if input_df.shape[0] == 0:
        st.warning("Please input the parameters to get the prediction.")
    else:
        if probability[0] < 0.5:
            st.info("Low likelihood of default.")
        else:
            st.error("The user may default next month.")
        
        # Calculate and Show Credit Utilization Ratio
        credit_utilization_ratio_percentage = credit_utilization_ratio * 100
        st.write(f"## Credit Utilization Ratio: {credit_utilization_ratio_percentage:.2f}%")

        # Show Probability of Default
        st.write(f"## Probability of Default: {probability[0]*100:.2f}%")

        # Info Button for Explanation
        if st.button("ℹ"):
            st.info("""
                The app utilizes a machine learning model to make predictions based on input values.
                When the app first runs, non-zero initial values are used to create the input DataFrame,
                which is then passed to the predict_default function.
                This function calculates a prediction and a probability of default based on the input data.
                Since the initial values are not all 0, the model makes a prediction based on these non-zero initial values.
                This leads to an initial prediction and probability being displayed even before the user has interacted with the input fields.
            """)

if __name__ == '__main__':
    main()