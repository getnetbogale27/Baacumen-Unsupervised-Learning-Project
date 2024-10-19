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





# Apply log transformation to Monetary (optional for reducing skewness)
rfm['Monetary'] = np.log1p(rfm['Monetary'])

# Standardize the RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Convert back to DataFrame
rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

# Elbow method to determine optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

# Create an expander for displaying the log-transformed and standardized data
with st.expander("🔢 RFM Data After Transformation and Scaling"):
    st.write("RFM Data after Log Transformation and Standardization:")
    st.dataframe(rfm_scaled)

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))  # Ensure all x-axis labels are shown
plt.grid()
st.pyplot(plt)  # Display the plot in Streamlit


