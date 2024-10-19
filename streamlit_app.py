import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/getnetbogale27/Baacumen-Unsupervised-Learning-Project/refs/heads/main/Dataset/Online_Retail_dataset.csv'
    df = pd.read_csv(url, encoding='ISO-8859-1')  # Use ISO encoding
    return df

# Load the data
df = load_data()

# Streamlit app title
st.title('🤖 Unsupervised ML App')
st.write("**Author:** Getnet B. (PhD Candidate)")
st.write("**Affiliation:** Baacumen Data Science Bootcamp")

st.info(
    "An online retailer seeks to understand its customers through transactional data. "
    "We used K-means clustering with the RFM (Recency, Frequency, Monetary) model to segment customers "
    "and uncover insights for targeted marketing."
)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# Remove rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])
# Filter out negative quantities or unit prices (returns or errors)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
# Create a TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
# Define the latest date to calculate recency
latest_date = df['InvoiceDate'].max()
# Create the RFM table
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency (count of unique invoices)
    'TotalPrice': 'sum'  # Monetary (total spending)
}).reset_index()
# Rename columns
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Merge the RFM data with the original dataset
df = df.merge(rfm, on='CustomerID', how='left')

# Expanders for different data views
with st.expander('🔢 Raw data (first 5 rows)'):
    st.write(df.head(5))  # Display first 5 rows of raw data with RFM columns

with st.expander('Data Types of Each Column'):
    st.write(df.dtypes)

with st.expander('🔢 RFM Data (first 5 rows)'):
    st.write(rfm.head(5))  # Display first 5 rows of RFM data

with st.expander('📊 View RFM Data Types'):
    st.write(rfm.dtypes)

# Select only numeric columns for scaling
dfAttr = df.select_dtypes(include=['float64', 'int64']).copy()

# Scale the data using z-score normalization
dfScaled = dfAttr.apply(zscore)

# Create an expander for the pairplot
with st.expander("📊 View Scaled Data Pairplot"):
    # Plot pairplot with KDE on diagonals
    sns.pairplot(dfScaled, diag_kind='kde')

    # Add title and improve layout
    plt.suptitle("Pairplot of Scaled Technical Support Data", y=1.02)
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(plt)
