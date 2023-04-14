import streamlit as st
import pandas as pd
import numpy as np
#import sklearn
import joblib
#fxn
#pipe_line=joblib.load(open('models/emotion_classifer2.pkl','rb'))
pipe_line=joblib.load(open('models/Emotion_classifer4.pkl','rb'))
#fxn
def predict_emotions(docx):
    results=pipe_line.predict([docx])
    return results[0]

def predict_emotions_probablity(docx):
    results=pipe_line.decision_function([docx])
    return results

def main():
    st.title('Consumer Emotion classifier app')
    menu=["Home","Monitor","About"]
    choice=st.sidebar.selectbox('Menu',menu)
    
    if choice == "Home":
        st.subheader('Home1-Emotion in text')
        
        with st.form(key='emotion clf form'):
            raw_text=st.text_area('Type Here')
            submit_text=st.form_submit_button(label='Submit')
        
        if submit_text:
            col1,col2 = st.columns(2)
            #apply fxn
            prediction=predict_emotions(raw_text)
            probability=predict_emotions_probablity(raw_text)
            
            with col1:
                st.success("original Text")
                st.write(raw_text)
                
                st.success('prediction')
                st.success(prediction)
            with col2:
                st.success("prediction probability")        
                st.write(probability)
            
    elif choice == "Monitor":        
        st.subheader('Monitor app')    

    else:
        st.subheader('About')

        
           
if __name__=="__main__":
    main()
    
        
        