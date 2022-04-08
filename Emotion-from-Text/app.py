import traceback
import streamlit as st
import altair as alt

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import joblib

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore') 

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

nltk.data.path.append('./nltk_data/')


max_seq_len = 500

stopword_all =  list(set(stopwords.words('english'))) 
stopword_all.extend(list(string.punctuation))


emoji_dict = {"anger":"😠", "fear":"😨😱", "love":"❤️🤗", "joy":"😂", "sadness":"😔","surprise":"😮"}


tokenizer = joblib.load('./model/tokenizer.pkl')
model = load_model('./model/lstm_model.h5')
label_encoder = joblib.load('./model/label_encoder.pkl')

def predict(text):
    seq = preprocess(text)
    res = model.predict(seq)
    return res

def preprocess(text):
    text = text.lower()
    text= re.sub(r"(#[\d\w\.]+)", '', text)
    text = re.sub(r"(@[\d\w\.]+)", '', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stopword_all]
    text = ' '.join(text)

    seq = tokenizer.texts_to_sequences([text])
    seq_pad = pad_sequences(seq,maxlen=max_seq_len)

    return seq_pad

def main():
    try:
        menu = ['Home','About']
        # choice = st.sidebar.selectbox("Menu",menu)

        st.header('Text Classification (Text-to-Emoji)')

        with st.form(key='text_to_emotion'):
            text_raw = st.text_area("Type Here !!")
            submit_btn = st.form_submit_button(label='Submit')
        if submit_btn:
            col1,col2 = st.columns(2)

            prob = predict(text_raw)

            col1.success("Original Text")
            col1.write(text_raw)
            
            col2.success("Prediction")
            res = label_encoder.inverse_transform([np.argmax(prob)])
            emoji = emoji_dict.get(res[0],'')
            col2.write("{} : {}".format(res[0],emoji))
            col2.write("Confidence : {}".format(np.max(prob)))
            
            st.success("Prediction probability")
            prob_df = pd.DataFrame(prob,columns=label_encoder.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ['emotions','probability']
            fig = alt.Chart(prob_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
            st.altair_chart(fig,use_container_width=True)

            st.write(prob_df_clean)

        st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)  
        col1, col2 = st.columns(2)
        col1.markdown(" By :- Himanshu Bag")
        col2.markdown(" Github Link for this project : [Click-Me](https://github.com/0x1h0b/Emotion-from-Text)")
        col1, col2 = st.columns(2)
        col1.markdown(" LinkedIn Profile : [himanshu-bag](https://www.linkedin.com/in/himanshu-bag/)")
        col2.markdown(" Github Profile : [0x1h0b](https://github.com/0x1h0b)")
    
    except Exception:
        print(traceback.print_exc())




if __name__ == "__main__":
    main()

# x="i am happy"
# raw = predict(x)
# print(raw)
# print(label_encoder.classes_)