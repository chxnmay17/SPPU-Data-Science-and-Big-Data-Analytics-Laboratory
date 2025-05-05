#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns

# Load the Iris dataset
df = sns.load_dataset("iris")

# Calculate mean, median, min, max, std for sepal_length grouped by species
summary_stats = df.groupby('species')['sepal_length'].agg(['mean', 'median', 'min', 'max', 'std'])


print("Summary Statistics Grouped by Species:\n")
print(summary_stats)


# In[5]:


# Display describe() for each species
for species in df['species'].unique():
    print(f"\nStatistics for {species}:")
    print(df[df['species'] == species].describe())


# In[ ]:




