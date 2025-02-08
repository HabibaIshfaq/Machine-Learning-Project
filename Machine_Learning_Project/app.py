from flask import Flask, request, jsonify
import pickle
import numpy as np
import streamlit as st
import pandas


# Load your trained random forest model
pickle_in = open('best_clf.pkl', 'rb')
best_clf = pickle.load(pickle_in)

def welcome():
    return 'welcome all'


def main():
    st.title("Diabetes Prediction Form")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Diabetes Prediction </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    gender = st.selectbox("Gender:", ["Male", "Female"])
    
    age = st.number_input("Age:", min_value=1)
    
    hypertension = st.selectbox("Hypertension:", ["No", "Yes"])
    
    heart_disease = st.selectbox("Heart Disease:", ["No", "Yes"])
    
    smoking_history = st.selectbox("Smoking History:", ["Never", "Former", "Current"])
    
    bmi = st.number_input("BMI:", min_value=0.0, format="%.1f")
    
    HbA1c_level = st.number_input("HbA1c Level:", min_value=0.0, format="%.1f")
    
    blood_glucose_level = st.number_input("Blood Glucose Level:", min_value=0)
    if gender=="Male":
        gender =1
    else:
        gender =0
    if hypertension=="Yes":
        hypertension =1
    else:
        hypertension =0
    if heart_disease=="Yes":
        heart_disease =1
    else:
        heart_disease =0
    if smoking_history.lower()=="no info":
        smoking_history =0
    elif smoking_history.lower()=="current":
        smoking_history =1
    elif smoking_history.lower()=="ever":
        smoking_history =2
    elif smoking_history.lower()=="former":
        smoking_history =3
    elif smoking_history.lower()=="never":
        smoking_history =4
    elif smoking_history.lower()=="not current":
        smoking_history =5


    
  
    result = ""
    if st.button("Submit"):
        result = best_clf.predict(pandas.DataFrame({"gender":[gender],"age":[age],"hypertension":[hypertension], "heart_disease":[heart_disease],"smoking_history":[smoking_history],"bmi":[bmi],"HbA1c_level": HbA1c_level,"blood_glucose_level": blood_glucose_level}))
        if result==0:
            result ="No"
        else:
            result ="Yes"
        st.write("The output is {}".format(result))

        # Here you would typically send the data to your backend or model for prediction

if __name__ == '__main__':
    main()
