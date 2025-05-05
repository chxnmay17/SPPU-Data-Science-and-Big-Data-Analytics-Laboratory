#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Import Required Python Libraries
import pandas as pd
import numpy as np
# Step 2: Load the Iris Dataset
# It's available online via seaborn or sklearn, or download the CSV directly.

# Option 1: Use seaborn to load
import seaborn as sns
df = sns.load_dataset("iris")

# Display first 5 rows
df.head()


# In[2]:


# Step 3: Check DataFrame Dimensions
print("Shape of the dataset:", df.shape)  # Rows and columns


# In[3]:


# Step 4: Data Preprocessing

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Descriptive statistics
print("\nDescriptive Stats:\n", df.describe())

# Info about data types and nulls
print("\nInfo:\n")
df.info()


# In[4]:


# Step 5: Data Formatting and Normalization

# Check data types
print("\nData Types Before:\n", df.dtypes)


# In[6]:


# Convert 'species' to categorical
df['species'] = df['species'].astype('category')
print("\nData Types After:\n", df.dtypes)


# In[7]:


# Normalize numeric columns
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = (
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] - df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].min()
) / (
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].max() - df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].min()
)


# In[8]:


df


# In[9]:


# Step 6: Convert Categorical Variables into Quantitative Variables

# One-hot encode the 'species' column
df_encoded = pd.get_dummies(df, columns=['species'], drop_first=True)

# Show encoded dataset
df_encoded.head()


# In[ ]:




