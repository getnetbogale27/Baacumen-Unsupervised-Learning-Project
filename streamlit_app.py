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


# Create an expander for the Distortion (Inertia) Plot
with st.expander("📉 Distortion Plot for Optimal Clusters"):
    # Plot the distortion values (WCSS/Inertia)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Distortion Plot (Inertia vs. Number of Clusters)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion (Inertia / WCSS)')
    plt.xticks(range(1, 11))  # Ensure all x-axis labels are shown
    plt.grid()

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Add a brief description
    st.info(
        "📌 **Distortion (Inertia)** measures how compact the clusters are. "
        "Lower values indicate that data points are closer to their cluster center. "
        "Look for the 'elbow point' where adding more clusters doesn't reduce distortion significantly."
    )




# Create an expander for the Elbow method plot
with st.expander("📈 Elbow Method for Optimal Clusters"):
    # Plot the Elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.xticks(range(1, 11))  # Ensure all x-axis labels are shown
    plt.grid()

    # Display the plot in Streamlit
    st.pyplot(plt)





from sklearn.metrics import silhouette_score

# Calculate Silhouette scores for 2 to 10 clusters
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(rfm_scaled)
    score = silhouette_score(rfm_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Create an expander for the Silhouette scores plot
with st.expander("📊 Silhouette Analysis for Optimal Clusters"):
    # Plot the Silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Score Analysis')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, 11))  # Ensure all x-axis labels are shown
    plt.grid()

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Show the optimal number of clusters based on the highest Silhouette score
    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
    st.write(f"📌 Optimal number of clusters based on Silhouette Score: **{optimal_clusters}**")








# Apply K-means with the chosen number of clusters (e.g., 2)
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Expander to preview RFM table with cluster assignments
with st.expander("📋 View RFM Table with Cluster Assignments"):
    st.write(rfm.head())  # Display the first few rows of RFM with clusters

# Expander for Scatter Plot: Frequency vs Monetary by Cluster
with st.expander("📊 Scatter Plot: Frequency vs Monetary by Cluster"):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Cluster', palette='viridis', s=60)
    plt.title('Customer Segments by Frequency and Monetary')
    plt.xlabel('Frequency')
    plt.ylabel('Monetary')
    plt.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(plt)

# Expander for Pairplot of RFM data by Cluster
with st.expander("📈 Pairplot of RFM Data by Cluster"):
    pairplot_fig = sns.pairplot(rfm, hue='Cluster', palette='coolwarm', diag_kind='kde', height=2.5)
    
    # Add title to pairplot
    pairplot_fig.fig.suptitle('Pairplot of RFM Data by Cluster', y=1.02)
    
    # Display the pairplot in Streamlit
    st.pyplot(pairplot_fig)


