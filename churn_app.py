import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl

st.sidebar.title('Classification model for Churn prediction')

CustomerID = int( st.number_input('CustomerID') )
Name = st.text_input('Name',placeholder='name')
Age = int(st.number_input('Age'))
Subscription_Length_Months = int( st.number_input('Subscription Length Months') )
Monthly_Bill = int( st.number_input('Monthly Bill') )
Total_Usage_GB = int( st.number_input('Total Usage GB') )
Gender = st.radio('Gender',['Male','Female'])
Location = st.radio('Location',['Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston'])

input_array = pd.DataFrame(np.array([CustomerID,Name,Age,Subscription_Length_Months,Monthly_Bill,Total_Usage_GB,Gender,Location]), index=None).T
input_array.columns = ['CustomerID','Name','Age','Subscription_Length_Months','Monthly_Bill','Total_Usage_GB','Gender','Location']


if st.button('Submit'):
    input_array=input_array.drop(columns=['Name','Total_Usage_GB'], axis=1)
    with open('Churn-prediction-App/onehotencoder.pkl','rb') as f:
        ohe = pkl.load(f)
    
    ohe_GL = ohe.transform(input_array[['Gender','Location']])

    input_array['Gender_Male'] = ohe_GL[:,0] # Gender_Male
    input_array['Location_Houston'] = ohe_GL[:,1] # Location_Houston
    input_array['Location_Los_Angeles'] = ohe_GL[:,1] # Location_Los_Angeles
    input_array['Location_Miami'] = ohe_GL[:,3] # Location_Miami
    input_array['Location_New_York'] = ohe_GL[:,4] # Location_New_York
    
    input_array = input_array.drop(columns=['Gender','Location'])

    with open('model.pkl', 'rb') as f:
        model = pkl.load(f)

    ans = int( model.predict(input_array) )
    if ans==1:
        st.write('Prediction is: ',ans, 'means the customer had churned' )
    else:
        st.write('Prediction is: ',ans, 'means the customer still using the service' )
        
    

