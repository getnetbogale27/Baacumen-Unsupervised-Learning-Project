import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/getnetbogale27/Baacumen-Unsupervised-Learning-Project/refs/heads/main/Dataset/Online_Retail_dataset.csv'
    df = pd.read_csv(url, encoding='ISO-8859-1')  # Use ISO encoding
    return df

# Load the data
df = load_data()

# Streamlit app title
st.title('🤖 Unsupervised Machine Learning App')

st.info(
    "An online retailer wants to understand its customer base and their purchasing patterns. "
    "The provided dataset contains transactional data, including purchase amounts, frequency, and customer information. "
    "Your objective is to segment the customers using K-means clustering based on their transaction behavior "
    "(e.g., purchase frequency, spending amount, and recency). Use the RFM (Recency, Frequency, Monetary) model as part of your clustering process "
    "and provide insights into customer loyalty and behavior for targeted marketing strategies."
)


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
