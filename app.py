import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle
import joblib
from model_to_production import *

with open("model/model.pkl", "rb") as file:
    model = pickle.load(file)

def main(): 
    st.markdown(f'''<p style="color: green;">note : the values in the dataset have been converted to meaningless symbols
                 to protect the confidentiality of the data and we do not know them.! <br>
                in real projects we know them in the deploy phase.</p>''', unsafe_allow_html=True)
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Predicting Credit Card Approvals </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    n0 = st.selectbox("0", ['b', 'a'])
    n1 = st.text_input("1","0") 
    n2 = st.text_input("2","0") 
    n3 = st.selectbox("3", ['u', 'y', 'l']) 
    n4 = st.selectbox("4",['g', 'p', '?', 'gg']) 
    n5 = st.selectbox("5",['w', 'q', 'm', 'r', 'cc', 'k', 'c', 'd', 'x', 'i', 'e', 'aa', 'ff','j']) 
    n6 = st.selectbox("6",['v', 'h', 'bb', 'ff', 'j', 'z', 'o', 'dd', 'n']) 
    n7 = n1 = st.text_input("7","0") 
    n8 = st.selectbox("8",['t', 'f']) 
    n9 = st.selectbox("9",['t', 'f']) 
    n10 = st.text_input("10","0") 
    n11 = st.selectbox("11",['t', 'f']) 
    n12 = st.selectbox("12",['g', 's', 'p']) 
    n13 = st.text_input("13","0") 
    n14 = st.text_input("14","0") 
    
    if st.button("Predict"): 
        features = [[n0,n1,float(n2),n3,n4,n5,n6,float(n7),n8,n9,int(n10),n11,n12,n13,int(n14)]]
        df=pd.DataFrame(features, columns=list(range(15)))
                
        prediction = model.predict(df)
        output = int(prediction[0])
        if output == 1:
            text = "Congratulations, approval!"
            st.markdown(f'<p style="color: green;">Approval status {text}</p>', unsafe_allow_html=True)
        else:
            text = "Sorry, not approved."
            st.markdown(f'<p style="color: red;">Approval status {text}</p>', unsafe_allow_html=True)
      
if __name__=='__main__': 
    main()