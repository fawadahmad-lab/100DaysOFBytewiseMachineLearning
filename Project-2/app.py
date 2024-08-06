import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

# Download necessary NLTK resources
nltk.download('punkt')

# Load the dataset
url = 'https://raw.githubusercontent.com/fawadahmad-lab/100DaysOFBytewiseMachineLearning/94d9763c96a8861916f7fdd2a202bdd8736d3429/Project2/Recomendation_sys/amazon_product.csv'
data = pd.read_csv(url)

# Remove unnecessary columns
data = data.drop('id', axis=1)

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')
def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Create stemmed tokens column
data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

# Define TF-IDF vectorizer and cosine similarity function
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english')
def cosine_sim(text1, text2):
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(text1), ' '.join(text2)])
    return cosine_similarity(tfidf_matrix)[0][1]

# Define search function
def search_products(query):
    query_stemmed = tokenize_and_stem(query)
    data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    results = data.sort_values(by=['similarity'], ascending=False).head(10)[['Title', 'Description', 'Category']]
    return results

# Web app
img = Image.open('img.jpg')
st.image(img, width=500)
st.title("Search Engine and Product Recommendation System on Amazon Data")
query = st.text_input("Enter Product Name")
submit = st.button('Search')
if submit:
    res = search_products(query)
    st.write(res)
