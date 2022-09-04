import numpy as np
import pickle
import pandas as pd


from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


rating_table = pickle.load(open("rating_table.pkl","rb"))


books_name = rating_table.index.to_list()


sparse_matrix = csr_matrix(rating_table)
model = NearestNeighbors(algorithm='brute')
model.fit(sparse_matrix)


#Function for recommending books
def recommend(book_name):
  recommended_books = []
  image_url = []
  book_index = np.where(rating_table.index==book_name)[0][0]
  distances , suggestions = model.kneighbors(rating_table.iloc[book_index,:].values.reshape(1,-1),n_neighbors=5)
  suggestions = np.ravel(suggestions, order='C') #2d to 1d array
  for i in suggestions:
    recommended_books.append(rating_table.index[i])

 
    
  return recommended_books