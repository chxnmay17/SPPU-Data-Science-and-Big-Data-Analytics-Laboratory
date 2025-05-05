#!/usr/bin/env python
# coding: utf-8

# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Display first few rows
print(titanic.head())

# Survival count by gender
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title('Survival by Gender')
plt.show()

# Survival count by class
sns.countplot(x='class', hue='survived', data=titanic)
plt.title('Survival by Passenger Class')
plt.show()

# Age distribution
sns.histplot(titanic['age'].dropna(), kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[7]:


# Fare distribution
sns.histplot(titanic['fare'], kde=True, bins=30)
plt.title('Ticket Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.show()


# In[ ]:




