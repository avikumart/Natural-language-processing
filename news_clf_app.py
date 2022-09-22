import numpy as np
import tensorflow as tf
from tensorflow import keras
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import wordcloud
from wordcloud import STOPWORDS
import sklearn
import streamlit as st

nltk.download('all')

# st components 
header = st.container()
intro = st.container()
data_info = st.container()
text_input = st.container()

with header:
    st.title("Welcome to News descriptions classification web application")
    st.text("This is the most simple text classification demo app")
    
with intro:
    st.subheader("Introduction to the projecrt")
    st.text("""Text is one of the most widespread forms of sequence data. 
It can be understood as either a sequence of charactors or a sequence of words, 
it's most common to work at level of words. Text-sequence processing 
includes following applications:
            
Applications of deep learning for text data:
            
1. Document classification
2. Articles lebelling
3. Sentiment analysis
4. Author identification 
5. Question-answering
6. Language detection
7. Translation Tasks
""")
    st.text("""Typical workflow to prepare text data for machine learning models:
1. Tokenization
2. One-Hot encoding or word indexing
3. Pad sequencing
4. Embedding layer (Word2Vec)
5. Corresponding word vector 
""")
    
with data_info:
    st.subheader("Dataset information and feature description")
    st.text("The dataset file contains 202,372 records. Each json record contains following attributes:")
    st.text("""category: Category article belongs to
headline: Headline of the article
authors: Person authored the article
link: Link to the post
short_description: Short description of the article
date: Date the article was published""")
    
with text_input:
    input_t = st.text_input("Enter news description here:", value=" ")

# clean the input sequences:
def data_cleaning(text):
    whitespace = re.compile(r"\s+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    text = whitespace.sub(' ', text)
    text = user.sub('', text)
    text = re.sub(r"\[[^()]*\]","", text)
    text = re.sub("\d+", "", text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", text)
    text = text.lower()
    # removing stop-words
    text = [word for word in text.split() if word not in list(STOPWORDS)]
    # word lemmatization
    sentence = []
    for word in text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word,'v'))    
    return ' '.join(sentence)
    
# load the dataset to tokenize the sentence
import pandas as pd
df = pd.read_json("News_Category_Dataset_v2.json", lines=True)
df['length_of_news'] = df['headline'] + df['short_description']
df.drop(['headline','short_description'], inplace=True, axis=1)
politics_list = list(df[df['category'] == 'POLITICS'].index) 
list_16 = politics_list[:16000] # list of 16000 row labels of POLITICS category
ndf = df.copy()
ndf.drop(list_16, axis=0, inplace=True)

# tokenize the input_t into integer list
tokenizer = keras.preprocessing.text.Tokenizer(num_words=100000, oov_token='<00V>')
tokenizer.fit_on_texts(ndf['length_of_news'])

# encode the labels
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cats = encoder.fit_transform(ndf['category'])
list_cat = list(encoder.classes_)
cat_dict = {}
for k, v in enumerate(list_cat):
    cat_dict.update({k:v})

# take input text, clean the text, tokenize it, pad it and feed it to the model
seq = data_cleaning(input_t)
token_seq = tokenizer.texts_to_sequences(seq)
token_pad = keras.preprocessing.sequence.pad_sequences(token_seq, maxlen=130)

# load the model for our prediction task
model  = tf.keras.models.load_model('new_clf_exp06.h5')

if input_t is None:
    st.text("Please entered any news description")
else:
    st.write("You have entered news description as following:\n", input_t)
    output = model.predict(token_pad)
    output2 = output[0]
    arg_max = np.argmax(output2)
    output_cat = cat_dict[arg_max]
    st.write(f"News category for your description {input_t} is {output_cat}")
