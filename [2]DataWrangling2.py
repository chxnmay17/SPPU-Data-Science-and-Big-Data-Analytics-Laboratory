#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Step 2: Create an "Academic Performance" Dataset with Skewed Data
np.random.seed(42)

data = {
    'Student_ID': range(1, 101),
    'Math_Score': np.random.exponential(scale=30, size=100),  # intentionally skewed
    'English_Score': np.random.normal(loc=70, scale=10, size=100)  # normally distributed
}

df = pd.DataFrame(data)

# Introduce some missing values
df.loc[[5, 12, 25], 'Math_Score'] = np.nan
df.loc[[3, 15], 'English_Score'] = np.nan

# Display  the data
df


# In[11]:


print("\nMissing values before filling:\n", df.isnull().sum())


# In[12]:


# Fill missing values with median (as requested)
df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].median())
df['English_Score'] = df['English_Score'].fillna(df['English_Score'].median())

print("\nMissing values after filling:\n", df.isnull().sum())


# In[15]:


# Z-score method to detect outliers in Math_Score
z_scores = stats.zscore(df['Math_Score'])
outliers = np.where(np.abs(z_scores) > 3)[0]

print("\nOutlier indices (Math_Score):", outliers[0])


# In[16]:


print("Outlier value in Math_Score:", df.loc[69, 'Math_Score'])


# In[17]:


# Replace outliers with median
median_math = df['Math_Score'].median()
df.loc[outliers, 'Math_Score'] = median_math


# In[18]:


print("Outlier value after:", df.loc[69, 'Math_Score'])


# In[19]:


sns.histplot(df['Math_Score'], kde=True)
plt.title("Original Math_Score Distribution")
plt.show()


# In[20]:


# Apply log transformation (add 1 to avoid log(0))
df['Math_Score_Log'] = np.log(df['Math_Score'] + 1)

# Histogram after transformation
sns.histplot(df['Math_Score_Log'], kde=True)
plt.title("Log-Transformed Math_Score Distribution")
plt.show()


# In[ ]:




