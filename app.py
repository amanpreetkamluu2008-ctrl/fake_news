import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import ssl
import nltk
import streamlit as st


df = pd.read_csv("train.csv")
df = df.fillna(" ")
df['content'] = df['author'] + " " + df['text']
X = df['content'].values
Y = df['label'].values 
# To handle SSL certificate verification issue during NLTK Download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
def stemming(content):
    content = str(content) #Ensure content is a string
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return  " ".join(stemmed_content)

# Apply stemming function
df['content']=df['content'].apply(stemming)

X = df['content'].values
Y = df['label'].values 
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

X_train, X_test, Y_train, Y_test =  train_test_split(X,Y, test_size = .2, random_state =42 )

model = LogisticRegression()
model.fit(X_train,Y_train)

# website

st.title("Fake news detection")
input_text = st.text_input("Enter news Article")

def prediction( input_text):
    input_text = stemming(input_text)
    input_data = vector.transform([input_text])
    prediction= model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:                           
        st.write("The news article is Real")
    else:
        st.write("The news article is Fake")
