#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Step 1: Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Step 3: Display first few rows (optional)
titanic.head()

# Step 4: Drop rows with missing 'age' or 'sex' values
titanic_clean = titanic.dropna(subset=['age', 'sex'])

# Custom colors
custom_palette = {0: 'red', 1: 'green'}

# Step 5: Create a boxplot for age distribution by sex and survival
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=titanic_clean,palette=custom_palette)
plt.title('Age Distribution by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.legend(title='Survived')
plt.grid(True)
plt.tight_layout()
plt.show()



# In[ ]:




