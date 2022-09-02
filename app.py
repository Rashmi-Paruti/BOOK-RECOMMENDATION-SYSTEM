# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""



import pandas as pd

import pickle
import streamlit as st
import numpy as np


# In[22]:


st.set_page_config(page_title="Book Recommendation System",)


# In[23]:


model=pickle.load(open('ModelRecommender.sav','rb'))


# In[44]:


final_rating=pd.read_csv('Ratings.csv')
book_pivot=final_rating.pivot_table(columns='User_id',index='Title',values='Rating')
book_list=book_pivot.index
book_pivot.fillna(0, inplace=True)


# In[45]:


books = [title for title in book_list]


# In[16]:


def recommend_book(book_name):
    book_id=np.where(book_pivot.index==book_name)[0][0]
    distances, suggetions = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)
    suggetions=suggetions.tolist()
    suggetions=suggetions[0]
    suggetions.pop(0)
    for i in range(len(suggetions)):
        st.success(book_pivot.index[suggetions[i]])


# In[19]:


def main():
    st.title("Book Recommendation System")
    select_book = st.selectbox('Select book: (Recommendation will be based on this selection)', ['--Select--'] + books)
    if select_book == '--Select--':
        st.warning('Please select Book!!')
    else:
        if st.button('Recommend book'):
            recommend_book(select_book)


# In[20]:


main()


# In[ ]: