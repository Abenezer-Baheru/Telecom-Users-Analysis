import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the cleaned data
data_path = '../src/data/cleaned_telecom_users_data.csv'
df = pd.read_csv(data_path)

# Task 1.1: Aggregate user behavior on applications
user_behavior = df.groupby('MSISDN/Number').agg({
    'Bearer Id': 'count',
    'Dur. (ms)': 'sum',
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum',
    'Social Media DL (Bytes)': 'sum',
    'Social Media UL (Bytes)': 'sum',
    'Google DL (Bytes)': 'sum',
    'Google UL (Bytes)': 'sum',
    'Email DL (Bytes)': 'sum',
    'Email UL (Bytes)': 'sum',
    'Youtube DL (Bytes)': 'sum',
    'Youtube UL (Bytes)': 'sum',
    'Netflix DL (Bytes)': 'sum',
    'Netflix UL (Bytes)': 'sum',
    'Gaming DL (Bytes)': 'sum',
    'Gaming UL (Bytes)': 'sum',
    'Other DL (Bytes)': 'sum',
    'Other UL (Bytes)': 'sum'
}).reset_index()

# Rename columns for clarity
user_behavior.columns = [
    'MSISDN/Number', 'Number of xDR Sessions', 'Total Session Duration (ms)', 
    'Total DL (Bytes)', 'Total UL (Bytes)', 'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
    'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)',
    'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
    'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)'
]

# Calculate the total data volume (in Bytes) for each application
user_behavior['Total Social Media Data (Bytes)'] = user_behavior['Social Media DL (Bytes)'] + user_behavior['Social Media UL (Bytes)']
user_behavior['Total Google Data (Bytes)'] = user_behavior['Google DL (Bytes)'] + user_behavior['Google UL (Bytes)']
user_behavior['Total Email Data (Bytes)'] = user_behavior['Email DL (Bytes)'] + user_behavior['Email UL (Bytes)']
user_behavior['Total Youtube Data (Bytes)'] = user_behavior['Youtube DL (Bytes)'] + user_behavior['Youtube UL (Bytes)']
user_behavior['Total Netflix Data (Bytes)'] = user_behavior['Netflix DL (Bytes)'] + user_behavior['Netflix UL (Bytes)']
user_behavior['Total Gaming Data (Bytes)'] = user_behavior['Gaming DL (Bytes)'] + user_behavior['Gaming UL (Bytes)']
user_behavior['Total Other Data (Bytes)'] = user_behavior['Other DL (Bytes)'] + user_behavior['Other UL (Bytes)']

# Task 1.2: Conduct an exploratory data analysis
# Describe all relevant variables and associated data types
print("Data Types:")
print(user_behavior.dtypes)

# Identify and treat missing values
user_behavior.replace('missing_bearer_Id', np.nan, inplace=True)
user_behavior.replace('missing_IMSI', np.nan, inplace=True)
user_behavior.replace('missing_MSISDN/Number', np.nan, inplace=True)
user_behavior.replace('missing_IMEI', np.nan, inplace=True)
user_behavior = user_behavior.apply(pd.to_numeric, errors='coerce')
user_behavior.fillna(user_behavior.mean(), inplace=True)

# Variable transformations
user_behavior['Decile Class'] = pd.qcut(user_behavior['Total Session Duration (ms)'], 5, labels=False)
user_behavior['Total Data (Bytes)'] = user_behavior['Total DL (Bytes)'] + user_behavior['Total UL (Bytes)']
total_data_per_decile = user_behavior.groupby('Decile Class')['Total Data (Bytes)'].sum()
print("Total Data (DL+UL) per Decile Class:")
print(total_data_per_decile)

# Analyze basic metrics
basic_metrics = user_behavior.describe()
print("Basic Metrics:")
print(basic_metrics)

# Non-Graphical Univariate Analysis
dispersion_parameters = user_behavior.var()
print("Dispersion Parameters (Variance):")
print(dispersion_parameters)
dispersion_parameters = user_behavior.std()
print("Dispersion Parameters (Standard Deviation):")
print(dispersion_parameters)

# Graphical Univariate Analysis
plt.figure(figsize=(15, 10))
plt.subplot(3, 3, 1)
sns.histplot(user_behavior['Number of xDR Sessions'], bins=30, kde=True)
plt.title('Number of xDR Sessions')
plt.subplot(3, 3, 2)
sns.histplot(user_behavior['Total Session Duration (ms)'], bins=30, kde=True)
plt.title('Total Session Duration (ms)')
plt.subplot(3, 3, 3)
sns.histplot(user_behavior['Total DL (Bytes)'], bins=30, kde=True)
plt.title('Total DL (Bytes)')
plt.subplot(3, 3, 4)
sns.histplot(user_behavior['Total UL (Bytes)'], bins=30, kde=True)
plt.title('Total UL (Bytes)')
plt.subplot(3, 3, 5)
sns.histplot(user_behavior['Total Social Media Data (Bytes)'], bins=30, kde=True)
plt.title('Total Social Media Data (Bytes)')
plt.subplot(3, 3, 6)
sns.histplot(user_behavior['Total Google Data (Bytes)'], bins=30, kde=True)
plt.title('Total Google Data (Bytes)')
plt.subplot(3, 3, 7)
sns.histplot(user_behavior['Total Email Data (Bytes)'], bins=30, kde=True)
plt.title('Total Email Data (Bytes)')
plt.subplot(3, 3, 8)
sns.histplot(user_behavior['Total Youtube Data (Bytes)'], bins=30, kde=True)
plt.title('Total Youtube Data (Bytes)')
plt.subplot(3, 3, 9)
sns.histplot(user_behavior['Total Netflix Data (Bytes)'], bins=30, kde=True)
plt.title('Total Netflix Data (Bytes)')
plt.tight_layout()
plt.show()

# Bivariate Analysis
plt.figure(figsize=(15, 10))
plt.subplot(3, 3, 1)
sns.scatterplot(x='Total Data (Bytes)', y='Total Social Media Data (Bytes)', data=user_behavior)
plt.title('Total Data vs. Social Media Data')
plt.subplot(3, 3, 2)
sns.scatterplot(x='Total Data (Bytes)', y='Total Google Data (Bytes)', data=user_behavior)
plt.title('Total Data vs. Google Data')
plt.subplot(3, 3, 3)
sns.scatterplot(x='Total Data (Bytes)', y='Total Email Data (Bytes)', data=user_behavior)
plt.title('Total Data vs. Email Data')
plt.subplot(3, 3, 4)
sns.scatterplot(x='Total Data (Bytes)', y='Total Youtube Data (Bytes)', data=user_behavior)
plt.title('Total Data vs. Youtube Data')
plt.subplot(3, 3, 5)
sns.scatterplot(x='Total Data (Bytes)', y='Total Netflix Data (Bytes)', data=user_behavior)
plt.title('Total Data vs. Netflix Data')
plt.subplot(3, 3, 6)
sns.scatterplot(x='Total Data (Bytes)', y='Total Gaming Data (Bytes)', data=user_behavior)
plt.title('Total Data vs. Gaming Data')
plt.subplot(3, 3, 7)
sns.scatterplot(x='Total Data (Bytes)', y='Total Other Data (Bytes)', data=user_behavior)
plt.title('Total Data vs. Other Data')
plt.tight_layout()
plt.show()

# Correlation Analysis
correlation_matrix = user_behavior[['Total Social Media Data (Bytes)', 'Total Google Data (Bytes)', 'Total Email Data (Bytes)', 
                                    'Total Youtube Data (Bytes)', 'Total Netflix Data (Bytes)', 'Total Gaming Data (Bytes)', 
                                    'Total Other Data (Bytes)']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
print("Correlation Matrix:")
print(correlation_matrix)

# Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(user_behavior[['Total Social Media Data (Bytes)', 'Total Google Data (Bytes)', 
                                                        'Total Email Data (Bytes)', 'Total Youtube Data (Bytes)', 
                                                        'Total Netflix Data (Bytes)', 'Total Gaming Data (Bytes)', 
                                                        'Total Other Data (Bytes)']])
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pca_df)
plt.title('PCA of Application Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:")
print(explained_variance_ratio)