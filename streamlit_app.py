#Step 1
# Imports and Data Loading
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Cache data loading
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/getnetbogale27/Baacumen-Unsupervised-Learning-Project/refs/heads/main/Dataset/Online_Retail_dataset.csv'
    df = pd.read_csv(url, encoding='ISO-8859-1')  # ISO encoding for special characters
    return df

# Load the dataset
df = load_data()


#Step 2
# Streamlit App Setup
# Streamlit title and info
st.title('🤖 Unsupervised ML App')
st.write("**Author:** Getnet B. (PhD Candidate)")

st.info(
    "Objective: An online retailer seeks to understand its customers through transactional data. "
    "Using K-means clustering with the RFM model, we segment customers and uncover insights for targeted marketing."
)




#Step 3
# Data Cleaning & RFM Calculation
# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Remove rows with missing CustomerID and negative values
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Create TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Calculate Recency, Frequency, and Monetary values
latest_date = df['InvoiceDate'].max()
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalPrice': 'sum'  # Monetary
}).reset_index()

# Rename columns for clarity
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Apply log transformation to Monetary for skewness correction
rfm['Recency'] = np.log1p(rfm['Recency'])
rfm['Frequency'] = np.sqrt(rfm['Frequency'])
rfm['Monetary'] = np.log1p(rfm['Monetary'])

# Standardize RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])



#Step 4
# Data Visualizations and Exploratory Analysis
# Display raw data and data types
with st.expander('🔢 Raw data (first 5 rows)'):
    st.write(df.head(5))
    st.info(
        "📌 **Note:**\n"
        "- **Total Dimension**: 541909\n"
        "- **Total Features**: 8"
    )

with st.expander('📋 Data Types'):
    st.write(df.dtypes)

# Pairplot of scaled data
with st.expander("📊 Pairplot of Scaled Data"):
    sns.pairplot(rfm_scaled, diag_kind='kde')
    plt.suptitle("Pairplot of Scaled RFM Data", y=1.02)
    st.pyplot(plt)
    plt.close()  
    st.info(
        "📌 **Remark:**\n"
        "- **Monetary vs. Frequency**: This plot indicates a positive relationship, suggesting that customers who spend more (higher monetary value) also tend to engage more frequently with the business. This correlation is significant as it highlights the importance of customer loyalty and spending patterns, indicating that targeting high-spending customers could yield further benefits.\n"
        "- **Recency vs. Other Metrics**: The relationships between **Recency** and the other variables (Monetary and Frequency) appear to be weak or negligible. This implies that recent customers do not necessarily correspond to high spending or high frequency, indicating a potential segment of customers who may need different marketing strategies to increase their engagement.\n"
        "- **Overall Observations**: The diagonal plots display the distribution of each variable, allowing for insights into their individual behaviors. The density plots indicate the spread and central tendency of each variable. Notably, any outliers or skewness in these distributions should be addressed, as they may influence the analysis and modeling processes.\n"
        "- **Implications for Strategy**: Understanding these relationships can inform targeted marketing strategies. For example, increasing engagement efforts on high-frequency, high-monetary customers could be a priority, while strategies to re-engage lower-frequency customers may also be considered."
    )



#Step 5
# Cluster Analysis: Elbow Method and Silhouette Score
# Elbow method for determining optimal clusters
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(rfm_scaled)
#     wcss.append(kmeans.inertia_)

# with st.expander("📉 Distortion Plot (Elbow Method)"):
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
#     plt.title('Elbow Method for Optimal Clusters')
#     plt.xlabel('Number of Clusters')
#     plt.ylabel('WCSS (Inertia)')
#     plt.xticks(range(1, 11))
#     plt.grid()
#     st.pyplot(plt)
#     plt.close()


# Calculate WCSS for different number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
with st.expander("📉 Distortion Plot (Elbow Method)"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')

    # Circle the second cluster (turning point)
    plt.scatter(2, wcss[1], color='red', s=100, label='Turning Point')
    circle = plt.Circle((2, wcss[1]), 100, color='red', fill=False, linewidth=2, linestyle='--')
    plt.gca().add_artist(circle)

    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Inertia)')
    plt.xticks(range(1, 11))
    plt.grid()
    plt.legend()
    
    st.pyplot(plt)
    plt.close()
    
    st.info(
        "📌 **Remark:**\n"
        "- The elbow point is observed at **Cluster 2**, indicating a significant reduction in WCSS. This suggests that 2 clusters are optimal for grouping the data effectively."
        "- For further confirmation, we also utilized **Silhouette Score Analysis** in the preceding steps to validate the optimal number of clusters."
    )


# Silhouette analysis to evaluate cluster quality
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(rfm_scaled)
    score = silhouette_score(rfm_scaled, kmeans.labels_)
    silhouette_scores.append(score)

with st.expander("📊 Silhouette Score Analysis"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Score Analysis')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, 11))
    plt.grid()
    st.pyplot(plt)
    plt.close()

    # Display the optimal number of clusters
    optimal_clusters = np.argmax(silhouette_scores) + 2  # Offset by 2
    st.write(f"📌 Optimal number of clusters: **{optimal_clusters}**")



#Step 6
# K-Means Clustering and Visualization
# User selects number of clusters via slider
num_clusters = st.slider("Select Number of Clusters:", min_value=2, max_value=10, value=optimal_clusters)

# Apply K-means with selected clusters
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Display RFM table with cluster assignments
with st.expander("📋 RFM Table with Cluster Assignments"):
    st.write(rfm.head())

# Scatter plot of Frequency vs Monetary by Cluster
with st.expander("📊 Scatter Plot: Frequency vs Monetary by Cluster"):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Cluster', palette='viridis', s=60)
    plt.title('Customer Segments by Frequency and Monetary')
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

# Pairplot of RFM data by Cluster
with st.expander("📈 Pairplot of RFM Data by Cluster"):
    pairplot_fig = sns.pairplot(rfm, hue='Cluster', palette='coolwarm', diag_kind='kde', height=2.5)
    pairplot_fig.fig.suptitle('Pairplot of RFM Data by Cluster', y=1.02)
    st.pyplot(pairplot_fig)
    plt.close()


# Boxplot
# Boxplot
# Boxplot
# Box plots for all RFM features grouped by clusters
with st.expander("📊 Box Plot Grouped by Clusters"):
    # Select RFM feature to plot (Recency, Frequency, or Monetary)
    feature = st.selectbox("Select RFM feature for box plot:", ['Recency', 'Frequency', 'Monetary'])

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=rfm, x='Cluster', y=feature, palette='Set2')
    plt.title(f'{feature} Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.grid(True)

    st.pyplot(plt)
    plt.close()




#Step 7
# Cluster Summary Statistics
# Summary statistics for each cluster
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).reset_index()

with st.expander("📊 Cluster Summary Statistics"):
    st.dataframe(cluster_summary.style.format(precision=2))
    st.info(
        "📌 **How to Interpret:**\n"
        "- **Recency**: Lower values indicate more recent purchases.\n"
        "- **Frequency**: Higher values indicate frequent purchases.\n"
        "- **Monetary**: Higher values represent higher spending customers."
    )

# Cluster distribution plot
with st.expander("📊 Cluster Distribution"):
    sns.countplot(data=rfm, x='Cluster', palette='muted')
    plt.title('Customer Distribution across Clusters')
    st.pyplot(plt)
    plt.close()
