#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_sklearn = load_iris()
iris = pd.DataFrame(data=iris_sklearn.data, columns=iris_sklearn.feature_names)
iris['species'] = iris_sklearn.target
iris['species'] = iris['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 1. List down the features and their types
print("\n1. Features and their types:\n")
print(iris.dtypes)
print("\nFeature Types Summary:")
print(iris.info())


# In[2]:


# 2. Histogram for each feature
print("\n2. Histograms for each feature:")
iris.iloc[:, :-1].hist(bins=15, figsize=(10, 6), layout=(2, 2), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Iris Features", fontsize=14)
plt.tight_layout()
plt.show()


# In[3]:


# 3. Boxplot for each feature
print("\n3. Boxplots for each feature:")
plt.figure(figsize=(12, 8))
for i, column in enumerate(iris.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data=iris, x='species', y=column)
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()


# In[4]:


# 4. Compare distributions and identify outliers
print("\n4. Inference:")
for col in iris.columns[:-1]:
    Q1 = iris[col].quantile(0.25)
    Q3 = iris[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = iris[(iris[col] < (Q1 - 1.5 * IQR)) | (iris[col] > (Q3 + 1.5 * IQR))]
    print(f"{col}: {len(outliers)} outlier(s) found.")

