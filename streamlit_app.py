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
st.title('🧠 Unsupervised ML App')
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

    st.info(
        "📌 **Remark:**\n"
        "- The **Silhouette Score** analysis indicates that the optimal number of clusters is **Cluster 2**, with a score of approximately **0.43**."
        "- Beyond Cluster 2, the Silhouette Score begins to decline, suggesting that additional clusters may not enhance the quality of the clustering."
        "- This reinforces our confidence in using 2 clusters for this dataset, as it provides a balance between separation and compactness of clusters."
    )


#Step 6
# K-Means Clustering and Visualization
# Slider for selecting the number of clusters

# Expander for additional information regarding cluster selection
with st.expander("📊 Cluster Selection Info"):
    num_clusters = st.slider("Select Number of Clusters:", min_value=2, max_value=10, value=optimal_clusters)
    st.info(
        "📌 **Cluster Selection:**\n"
        "- We can use the slider above to select the number of clusters for the K-Means clustering analysis. \n"
        "- The default value is set to **Cluster 2**, based on our earlier analysis using both the **Elbow Method** and **Silhouette Score Analysis**."
    )


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
    st.info(
        "📌 **Scatter Plot Insights:**\n"
        "- The scatter plot visualizes customer segments based on their **Frequency** (number of purchases) and **Monetary** (total spending).\n"
        "- Customers in **Cluster 0** show the lowest frequency and low monetary value, indicating infrequent, low-value purchases.\n"
        "- In contrast, **Cluster 1** reveal relatively high frequency and monetary value, suggesting these customers are more engaged and tend to make higher-value purchases."
    )




# Pairplot of RFM data by Cluster
with st.expander("📈 Pairplot of RFM Data by Cluster"):
    pairplot_fig = sns.pairplot(rfm, hue='Cluster', palette='coolwarm', diag_kind='kde', height=2.5)
    pairplot_fig.fig.suptitle('Pairplot of RFM Data by Cluster', y=1.02)
    st.pyplot(pairplot_fig)
    plt.close()
    st.info(
        "📌 **Pairplot Insights:**\n"
        "- The pairplot provides a comprehensive view of the relationships between **Recency**, **Frequency**, and **Monetary** across different customer clusters.\n"
        "- Each cluster is represented in distinct colors, allowing for easy identificatiion of customer segments.\n"
        "- Notably, there is a visible separation bettween the clusters in the pairwise plots, indicating that different customer segments exhibit unique purchasing behaviors."
    )


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
    st.info(
        "📌 **Box Plot Insights:**\n"
        "- The box plot visualizes the distribution of the selected RFM feature across different customer clusters.\n"
        "- Each box represents the interquartile range (IQR), with the line inside the box indicating the median value for each cluster.\n"
        "- Outliers are displayed as individual points beyond the whiskers, providing insights into extreme values within each cluster.\n"
        "- For **Recency**, we may observe that certain clusters have significantly lower medians, indicating differences in customer engagement.\n"
        "- **Frequency** can reveal patterns of purchase behavior, showing which clusters have the most loyal customers.\n"
        "- The **Monetary** distribution highlights how much different clusters spend on average, allowing for targeted marketing strategies based on spending behavior."
    )



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
        "📌 **Cluster Summary Insights:**\n"
        "- **Cluster 0:**\n"
        f"  - **Average Recency:** {cluster_summary['Recency'].iloc[0]:.2f} days (indicating a more recent purchasing behavior).\n"
        f"  - **Average Frequency:** {cluster_summary['Frequency'].iloc[0]:.2f} purchases per customer (suggesting lower purchase frequency).\n"
        f"  - **Average Monetary:** ${cluster_summary['Monetary'].iloc[0]:.2f} (indicating lower spending).\n"
        f"  - **Customer Count:** {cluster_summary['CustomerID'].iloc[0]} customers in this cluster.\n\n"
        "- **Cluster 1:**\n"
        f"  - **Average Recency:** {cluster_summary['Recency'].iloc[1]:.2f} days (showing more recent purchases compared to Cluster 0).\n"
        f"  - **Average Frequency:** {cluster_summary['Frequency'].iloc[1]:.2f} purchases per customer (higher purchase frequency).\n"
        f"  - **Average Monetary:** ${cluster_summary['Monetary'].iloc[1]:.2f} (indicating higher spending).\n"
        f"  - **Customer Count:** {cluster_summary['CustomerID'].iloc[1]} customers in this cluster.\n\n"
        "📌 **Interpretation:**\n"
        "- **Recency:** Lower values indicate more recent purchases, suggesting engagement.\n"
        "- **Frequency:** Higher values indicate frequent purchases, pointing to loyal customers.\n"
        "- **Monetary:** Higher values represent customers who spend more, valuable for revenue.\n"
        "- These insights can guide marketing strategies, with a focus on retaining high-frequency, high-monetary customers in Cluster 1."
    )


#Step 8
# Cluster distribution plot
with st.expander("📊 Cluster Distribution"):
    sns.countplot(data=rfm, x='Cluster', palette='muted')
    plt.title('Customer Distribution across Clusters')
    st.pyplot(plt)
    plt.close()
    st.info(
        "📌 **Cluster Distribution Insights:**\n"
        "- The count plot displays the number of customers within each cluster, indicating the size of each customer segment.\n"
        "- **Cluster 0:** Contains the majority of customers, suggesting it may represent a larger group with common characteristics, likely lower spending and frequency.\n"
        "- **Cluster 1:** Has fewer customers compared to Cluster 0, indicating this group may consist of high-value customers who are more engaged and spend more.\n"
        "- Understanding the distribution is crucial for targeted marketing efforts and resource allocation, focusing on both high-volume (Cluster 0) and high-value (Cluster 1) segments."
    )

#Step 9
# Export the RFM DataFrame to a CSV file
rfm.to_csv('customer_segments.csv', index=False)

#Step 10
with st.expander("📈 Recommendations and Business Insights"):
    # Recommendations Section
    st.info(
        "### Recommendations:\n"
        "- **Targeted Marketing Campaigns:**\n"
        "  - **Cluster 0 (Low Frequency, Low Monetary):** Offer incentives such as discounts to encourage more frequent purchases.\n"
        "  - **Cluster 1 (High Frequency, High Monetary):** Tailor exclusive offers and VIP events to maintain loyalty and increase spending.\n\n"
        
        "- **Personalized Communication:**\n"
        "  - Use RFM insights to send targeted email campaigns that highlight products related to previous purchases.\n\n"

        "- **Customer Retention Programs:**\n"
        "  - Implement loyalty programs that reward frequent buyers and consider tiered loyalty levels to incentivize higher spending.\n\n"

        "- **Product Recommendations:**\n"
        "  - For Cluster 0, recommend complementary products to encourage additional spending.\n\n"

        "- **Monitoring and Feedback:**\n"
        "  - Continuously monitor customer behavior post-intervention and solicit feedback to adapt strategies.\n\n"

        "- **Customer Education:**\n"
        "  - Provide resources to educate customers on the value of higher spending products to encourage purchases.\n\n"

        "- **Upselling Opportunities:**\n"
        "  - Identify opportunities to upsell to Cluster 1 customers by highlighting premium products that align with their interests.\n\n"

        "- **Segment-Specific Promotions:**\n"
        "  - Design tailored promotions for each cluster, such as introductory offers for Cluster 0 and loyalty discounts for Cluster 1."
    )

    # Business Insights Section
    st.info(
        "### Business Insights:\n"
        "- **Understanding Customer Segmentation:** RFM analysis provides a clear picture of customer behavior, allowing effective audience segmentation.\n\n"
        "- **Data-Driven Decision Making:** Leveraging insights enables informed decisions regarding marketing strategies and customer engagement.\n\n"
        "- **Resource Allocation:** Focus marketing budgets on high-value clusters to maximize ROI and enhance overall profitability."
    )
