

import pandas as pd
import pickle
import streamlit as st
import numpy as np

st.set_page_config(page_title="Book Recommendation System",)
model=pickle.load(open('final_model.pkl','rb'))
