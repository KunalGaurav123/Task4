import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data creation
np.random.seed(42)  # For reproducibility

# Creating a sample dataset
data = {
    'Age': np.random.randint(18, 70, size=100),
    'Salary': np.random.randint(30000, 120000, size=100),
    'Gender': np.random.choice(['Male', 'Female'], size=100),
    'Department': np.random.choice(['HR', 'Engineering', 'Marketing', 'Sales'], size=100)
}

df = pd.DataFrame(data)

# Display basic information about the dataset
print(df.info())
print(df.describe())

# 1. Distribution of Variables

# Histograms for numerical variables
df.hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.suptitle('Histograms of Numerical Variables')
plt.show()

# Box plots for numerical variables
df.plot(kind='box', subplots=True, layout=(1, 2), figsize=(15, 6), sharex=False, sharey=False)
plt.suptitle('Box Plots of Numerical Variables')
plt.show()

# Bar charts for categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=df)
    plt.title(f'Count Plot of {col}')
    plt.show()

# 2. Identify Outliers

# Box plot to identify outliers
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    plt.show()

# 3. Check for Correlations

# Correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
