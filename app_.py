# -*- coding: utf-8 -*-
"""app_

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12u8oOuCBD5-S-kkMXCVoUJ3e-bAPi-c3
"""

import streamlit as st
import pickle
import pandas as pd
import requests

st.title('books Recommender System')

matrix = pickle.load(open("books.pk1","rb"))
model = pickle.load(open("model.pk1","rb"))
result = pickle.load(open("result.pk1","rb"))



def recommend(books):
    recommended_book_names = []
    distances, indices = model.kneighbors(matrix.loc[books].values.reshape(1, -1), n_neighbors=10)
    print("\nRecommended books:\n")
    for i in range(0, len(distances.flatten())):
        if i > 0:

           recommended_book_names.append(matrix.index[indices.flatten()[i]])
    return recommended_book_names


user_list = result['User-ID'].values
User_ID_book= st.selectbox('Select a books from drop down', user_list)
selected_user_id=result[result["User-ID"]==User_ID_book].sort_values('Book-Rating', ascending=False).head(1)
selected_book=selected_user_id['Book-Title'].values

#books_list = result['Book-Title'].values
#selected_book = st.selectbox('Select a books from drop down', books_list)

st.write('You selected:', User_ID_book)

if st.button('Show Recommend book'):
    recommended_book = recommend(selected_book)
    col1,col2,col3,col4,col5 = st.columns(5)

    with col1:
        st.text(recommended_book[0])

    with col2:
        st.text(recommended_book[1])

    with col3:
        st.text(recommended_book[2])

    with col4:
        st.text(recommended_book[3])

    with col5:
        st.text(recommended_book[4])
