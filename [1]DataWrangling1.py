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



EXPLAIN***********************************************************************************************************
    Sure! Here's a **line-by-line explanation** of your Jupyter Notebook code which processes the **Iris dataset** and prepares it for analysis or machine learning.

---

### **🔧 In\[1]: Import & Load Dataset**

```python
import pandas as pd
import numpy as np
```

* `pandas`: For handling data in tabular format (like spreadsheets).
* `numpy`: For numerical operations.

```python
import seaborn as sns
df = sns.load_dataset("iris")
```

* `seaborn`: A visualization library that also has built-in datasets.
* `load_dataset("iris")`: Loads the classic Iris flower dataset into a DataFrame `df`.

```python
df.head()
```

* Shows the **first 5 rows** of the dataset.

---

### **📏 In\[2]: Check Dimensions**

```python
print("Shape of the dataset:", df.shape)
```

* Displays number of rows and columns (should be 150 rows × 5 columns for Iris).

---

### **🧹 In\[3]: Data Cleaning**

```python
print("\nMissing Values:\n", df.isnull().sum())
```

* Checks for missing values in each column.

```python
print("\nDescriptive Stats:\n", df.describe())
```

* Shows statistics like mean, min, max, std dev for numerical columns.

```python
print("\nInfo:\n")
df.info()
```

* Displays the structure of the dataset: column types and null count.

---

### **🔄 In\[4] & In\[6]: Data Formatting**

```python
print("\nData Types Before:\n", df.dtypes)
```

* Lists current data types of each column.

```python
df['species'] = df['species'].astype('category')
print("\nData Types After:\n", df.dtypes)
```

* Converts `species` column to `category` type (which is more efficient for non-numeric labels).

---

### **📊 In\[7]: Normalize Data**

```python
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = (
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] - df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].min()
) / (
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].max() - df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].min()
)
```

* Performs **min-max normalization** to scale numeric values between **0 and 1**.
* Formula:

  $$
  \text{Normalized} = \frac{(x - \text{min})}{\text{max} - \text{min}}
  $$

---

### **📋 In\[8]: View Normalized Data**

```python
df
```

* Displays the entire DataFrame with normalized values.

---

### **🔁 In\[9]: One-Hot Encoding**

```python
df_encoded = pd.get_dummies(df, columns=['species'], drop_first=True)
```

* Converts `species` column into **binary columns** (e.g., species\_versicolor, species\_virginica).
* `drop_first=True` avoids multicollinearity by removing the first dummy column.

```python
df_encoded.head()
```

* Displays the first 5 rows of the encoded dataset.

---

Would you like a plot or model (e.g., classification or clustering) added to this cleaned dataset?

    

