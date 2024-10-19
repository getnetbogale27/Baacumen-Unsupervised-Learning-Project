import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/getnetbogale27/Baacumen-Unsupervised-Learning-Project/refs/heads/main/Dataset/Online_Retail_dataset.csv?token=GHSAT0AAAAAACY5XTINOJINW4RIB7MMIHMWZYRHK6Q'
    df = pd.read_csv(url)
    return df


# Load the data
df = load_data()

# Streamlit app title
st.title('🤖 Machine Learning App')

st.info('This app explores child undernutrition data and builds a predictive machine learning model.')

# Expanders for different data views
with st.expander('🔢 Raw data (first 5 rows)'):
    st.write(df.head(5))  # Display first 5 rows of raw data

with st.expander('🧩 X (independent variables) (first 5 rows)'):
    X_raw = df.iloc[:, 3:-1]
    st.write(X_raw.head(5))  # Display first 5 rows of independent variables

with st.expander('🎯 Y (dependent variable) (first 5 rows)'):
    y_raw = df.iloc[:, -1]
    # Display first 5 rows of dependent variable
    st.write(y_raw.head(5).reset_index(drop=True))
