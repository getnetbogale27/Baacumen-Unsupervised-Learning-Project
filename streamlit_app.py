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

st.info(
    "Objective: An online retailer seeks to understand its customers through transactional data. "
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

# Scale the data using z-score normalization
dfScaled = rfm[['Recency', 'Frequency', 'Monetary']].apply(zscore)

# User input for number of clusters
num_clusters = st.slider("Select Number of Clusters:", min_value=2, max_value=10, value=4)

# K-Means Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
rfm['Cluster'] = kmeans.fit_predict(dfScaled)

# Elbow Method to visualize optimal clusters
sse = []
for i in range(1, 11):
    kmeans_temp = KMeans(n_clusters=i, random_state=0)
    kmeans_temp.fit(dfScaled)
    sse.append(kmeans_temp.inertia_)

# Plot the elbow graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
st.pyplot(plt)

# Expanders for different data views
with st.expander('🔢 Raw data (first 5 rows)'):
    st.write(rfm.head(5))  # Display first 5 rows of raw data with RFM columns

with st.expander('Data Types of Each Column'):
    st.write(rfm.dtypes)

with st.expander("📊 View Scaled Data Pairplot"):
    # Plot pairplot with KDE on diagonals
    sns.pairplot(dfScaled, diag_kind='kde')
    plt.suptitle("Pairplot of Scaled RFM Data", y=1.02)
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(plt)

# Display cluster summary
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).reset_index()

st.write("### Cluster Summary:")
st.write(cluster_summary)

# Visualize cluster distributions
sns.countplot(data=rfm, x='Cluster')
plt.title('Customer Distribution across Clusters')
st.pyplot(plt)
