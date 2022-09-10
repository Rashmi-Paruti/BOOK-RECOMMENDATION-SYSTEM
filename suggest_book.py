# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 21:03:45 2022

@author: rashm
"""

import pandas as pd
import streamlit as st
import pickle

similarity = pickle.load(open('C:/Users/rashm/Desktop/ExcerlR Python/Project/similarity.pkl', 'rb'))
df = pickle.load(open('C:/Users/rashm/Desktop/ExcerlR Python/Project/df.pkl', 'rb'))
books_data = pickle.load(open('C:/Users/rashm/Desktop/ExcerlR Python/Project/data.pkl', 'rb'))
data = pd.DataFrame(books_data)

def get_recommendation(user_index):
    idx = user_index
    sim_scores = list(enumerate(similarity[idx]))

    # get books that are unrated by the given user
    unrated_books = df.iloc[idx][df.iloc[idx].isna()].index

    # get weighted ratings of unrated books by all other users
    book_ratings = (df[unrated_books].T * similarity[idx]).T

    # get top 100 similar users by skipping the current user
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:101]

    # get mean of book ratings by top 100 most similar users for the unrated books
    book_ratings = book_ratings.iloc[[x[0] for x in sim_scores]].mean()
    
    # get rid of null values and sort it based on ratings
    book_ratings = book_ratings.reset_index().dropna().sort_values(0, ascending=False).iloc[:10]
    
    # get recommended book titles in sorted order
    recommended_books = data[data['ISBN'].isin(book_ratings['ISBN'])][['ISBN', 'Book-Title']]
    recommended_books = recommended_books.drop_duplicates('ISBN').reset_index(drop=True)

    return pd.DataFrame({'ISBN':recommended_books['ISBN'], 
                         'Recommended Book':recommended_books['Book-Title']})
    
def main():
    # giving a title
    st.title('Book Recommendation')
    # getting the input data from the user
    User_ID = st.text_input('User_ID')
    # code for Prediction
    recommendation = ''
    # creating a button for Prediction
    if st.button('Book recomendation'):
        recommendation = get_recommendation(User_ID)
        st.success(recommendation)
        


    
    