import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data_path = '../src/data/cleanedTelecomUsersData.csv'
df = pd.read_csv(data_path)

# Calculate average throughput and average TCP retransmission
df['Avg Throughput (MB)'] = (df['Avg Bearer TP DL (kbps)'] + df['Avg Bearer TP UL (kbps)']) / 2 / 1024
df['Avg TCP Retrans (MB)'] = df['TCP DL Retrans. Vol (Bytes)'] / (1024 ** 2)

# Perform k-means clustering for engagement score (using average throughput)
features_engagement = df[['Avg Throughput (MB)']]
kmeans_engagement = KMeans(n_clusters=3, random_state=42)
df['Cluster_Engagement'] = kmeans_engagement.fit_predict(features_engagement)

# Perform k-means clustering for experience score (using average TCP retransmission)
features_experience = df[['Avg TCP Retrans (MB)']]
kmeans_experience = KMeans(n_clusters=3, random_state=42)
df['Cluster_Experience'] = kmeans_experience.fit_predict(features_experience)

# Identify the less engaged cluster (cluster with the lowest average throughput)
less_engaged_cluster_center = kmeans_engagement.cluster_centers_[df.groupby('Cluster_Engagement')['Avg Throughput (MB)'].mean().idxmin()]

# Identify the worst experience cluster (cluster with the highest average TCP retransmission)
worst_experience_cluster_center = kmeans_experience.cluster_centers_[df.groupby('Cluster_Experience')['Avg TCP Retrans (MB)'].mean().idxmax()]

# Calculate engagement score (Euclidean distance to the less engaged cluster center)
df['Engagement_Score'] = df.apply(lambda row: euclidean(row[['Avg Throughput (MB)']], less_engaged_cluster_center), axis=1)

# Calculate experience score (Euclidean distance to the worst experience cluster center)
df['Experience_Score'] = df.apply(lambda row: euclidean(row[['Avg TCP Retrans (MB)']], worst_experience_cluster_center), axis=1)

# Display the first few rows of the dataframe with the new scores
print(df[['Bearer Id', 'Engagement_Score', 'Experience_Score']].head())

# Plotting the engagement scores
plt.figure(figsize=(12, 6))
sns.histplot(df['Engagement_Score'], bins=50, kde=True)
plt.title('Distribution of Engagement Scores', fontsize=20)
plt.xlabel('Engagement Score', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.show()

# Plotting the experience scores
plt.figure(figsize=(12, 6))
sns.histplot(df['Experience_Score'], bins=50, kde=True)
plt.title('Distribution of Experience Scores', fontsize=20)
plt.xlabel('Experience Score', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.show()

# Calculate satisfaction score as the average of engagement and experience scores
df['Satisfaction_Score'] = (df['Engagement_Score'] + df['Experience_Score']) / 2

# Report the top 10 satisfied customers
top_10_satisfied_customers = df[['Bearer Id', 'Satisfaction_Score']].sort_values(by='Satisfaction_Score', ascending=True).head(10)

# Display the top 10 satisfied customers
print(top_10_satisfied_customers)

# Plotting the top 10 satisfied customers
plt.figure(figsize=(12, 6))
sns.barplot(x='Bearer Id', y='Satisfaction_Score', data=top_10_satisfied_customers)
plt.title('Top 10 Satisfied Customers', fontsize=20)
plt.xlabel('Bearer Id', fontsize=16)
plt.ylabel('Satisfaction Score', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Normalize the engagement and experience scores
df['Engagement_Score_Norm'] = (df['Engagement_Score'] - df['Engagement_Score'].min()) / (df['Engagement_Score'].max() - df['Engagement_Score'].min())
df['Experience_Score_Norm'] = (df['Experience_Score'] - df['Experience_Score'].min()) / (df['Experience_Score'].max() - df['Experience_Score'].min())

# Calculate satisfaction score as the average of normalized engagement and experience scores
df['Satisfaction_Score'] = (df['Engagement_Score_Norm'] + df['Experience_Score_Norm']) / 2

# Report the top 10 satisfied customers
top_10_satisfied_customers = df[['Bearer Id', 'Satisfaction_Score']].sort_values(by='Satisfaction_Score', ascending=True).head(10)

# Display the top 10 satisfied customers
print(top_10_satisfied_customers)

# Plotting the top 10 satisfied customers
plt.figure(figsize=(12, 6))
sns.barplot(x='Bearer Id', y='Satisfaction_Score', data=top_10_satisfied_customers)
plt.title('Top 10 Satisfied Customers', fontsize=20)
plt.xlabel('Bearer Id', fontsize=16)
plt.ylabel('Satisfaction Score', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Select relevant features for the regression model
features = ['Avg Throughput (MB)', 'Avg TCP Retrans (MB)', 'Engagement_Score', 'Experience_Score']
X = df[features]
y = df['Satisfaction_Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Display the model coefficients
coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print(coefficients)

# Generate the linear equation string
linear_equation = f"y = {model.intercept_:.2f}"
for i, coef in enumerate(model.coef_):
    linear_equation += f" + {coef:.2f}*x{i+1}"

# Plotting the predicted vs actual satisfaction scores
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Predicted vs Actual Satisfaction Scores', fontsize=20)
plt.xlabel('Actual Satisfaction Score', fontsize=16)
plt.ylabel('Predicted Satisfaction Score', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)

# Add text for R-squared, MSE, and linear equation
plt.text(0.05, 0.95, f'R-squared: {r2:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
plt.text(0.05, 0.90, f'MSE: {mse:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
plt.text(0.05, 0.85, f'Linear Equation: {linear_equation}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

plt.tight_layout()
plt.show()

# Select the engagement and experience scores for clustering
features_clustering = df[['Engagement_Score', 'Experience_Score']]

# Perform k-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_clustering)

# Display the cluster centers
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Engagement_Score', 'Experience_Score'])
print(cluster_centers)

# Display the first few rows of the dataframe with the cluster labels
print(df[['Bearer Id', 'Engagement_Score', 'Experience_Score', 'Cluster']].head())

# Perform k-means clustering with k=2 for engagement score
kmeans_engagement = KMeans(n_clusters=2, random_state=42)
df['Cluster_Engagement'] = kmeans_engagement.fit_predict(df[['Engagement_Score']])

# Count the number of data points in each cluster
cluster_counts = df['Cluster'].value_counts()
engagement_cluster_counts = df['Cluster_Engagement'].value_counts()

# Plotting the clustering results for engagement and experience scores
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df, x='Engagement_Score', y='Experience_Score', hue='Cluster', palette='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
for i, (x, y) in enumerate(kmeans.cluster_centers_):
    plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=14, color='black', ha='right', fontweight='bold')
plt.title('K-Means Clustering (k=2) on Engagement and Experience Scores', fontsize=20)
plt.legend(title=f'Experience Cluster Counts: {cluster_counts.to_dict()}\nEngagement Cluster Counts: {engagement_cluster_counts.to_dict()}')
plt.tight_layout()
plt.show()

# Select the satisfaction and experience scores for clustering
features_clustering = df[['Satisfaction_Score', 'Experience_Score']]

# Perform k-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_clustering)

# Aggregate the average satisfaction and experience scores per cluster
average_scores_per_cluster = df.groupby('Cluster')[['Satisfaction_Score', 'Experience_Score']].mean().reset_index()
print(average_scores_per_cluster)

# Plotting the average satisfaction and experience scores per cluster
plt.figure(figsize=(6, 4))
average_scores_per_cluster.plot(kind='bar', x='Cluster', ax=plt.gca())
plt.title('Average Satisfaction and Experience Scores per Cluster', fontsize=20)
plt.xlabel('Cluster', fontsize=16)
plt.ylabel('Average Score', fontsize=16)
plt.xticks(rotation=0)
plt.legend(title='Scores', fontsize=12, title_fontsize=14)
plt.tight_layout()
plt.show()