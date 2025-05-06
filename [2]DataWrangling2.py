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

EXPLAIN********************************************************************************************************************8
Great! Here's a **detailed word-by-word breakdown** of the code, focusing on **what each word, symbol, and function means**:

---

### 1. **Shebang and Encoding**

```python
#!/usr/bin/env python
```

* `#!` – Shebang, used to tell the OS how to execute the file.
* `/usr/bin/env` – Searches for the Python interpreter in your environment’s `PATH`.
* `python` – The name of the interpreter to run the code.

```python
# coding: utf-8
```

* `#` – Comment symbol (ignored by Python).
* `coding:` – Declares the source file's text encoding.
* `utf-8` – A character encoding that supports all characters (Unicode).

---

### 2. **Import Libraries**

```python
import pandas as pd
```

* `import` – Python keyword to load external modules.
* `pandas` – A library for data analysis and manipulation.
* `as` – Keyword to assign an alias.
* `pd` – Alias used to refer to `pandas`.

```python
import numpy as np
```

* `numpy` – Library for numerical operations.
* `np` – Alias for `numpy`.

```python
import seaborn as sns
```

* `seaborn` – Library for advanced data visualization.
* `sns` – Alias for `seaborn`.

```python
import matplotlib.pyplot as plt
```

* `matplotlib.pyplot` – Plotting module of matplotlib.
* `plt` – Alias for plotting commands.

```python
from scipy import stats
```

* `from` – Keyword to import specific components from a module.
* `scipy` – Library for scientific computing.
* `import` – Imports part of a module.
* `stats` – Submodule for statistics.

---

### 3. **Set Random Seed**

```python
np.random.seed(42)
```

* `np` – Refers to `numpy`.
* `.random` – Random number generator.
* `.seed(42)` – Fixes the random output to be reproducible. `42` is the seed value.

---

### 4. **Create Data Dictionary**

```python
data = {
    'Student_ID': range(1, 101),
```

* `data` – Variable name for the dictionary.
* `=` – Assignment operator.
* `{` – Starts a dictionary.
* `'Student_ID'` – Key (column name).
* `range(1, 101)` – Generates numbers from 1 to 100.

```python
    'Math_Score': np.random.exponential(scale=30, size=100),
```

* `'Math_Score'` – Dictionary key.
* `np.random.exponential(...)` – Generates skewed (exponential) numbers.
* `scale=30` – Mean of the exponential distribution.
* `size=100` – Generates 100 values.

```python
    'English_Score': np.random.normal(loc=70, scale=10, size=100)
}
```

* `'English_Score'` – Key for the second subject.
* `np.random.normal(...)` – Generates numbers from a normal (bell curve) distribution.
* `loc=70` – Mean of the distribution.
* `scale=10` – Standard deviation (spread).
* `size=100` – Generates 100 values.
* `}` – Ends the dictionary.

---

### 5. **Create DataFrame**

```python
df = pd.DataFrame(data)
```

* `df` – Variable name for the DataFrame.
* `pd` – Refers to pandas.
* `.DataFrame(...)` – Converts the dictionary into a structured table.

---

### 6. **Add Missing Values**

```python
df.loc[[5, 12, 25], 'Math_Score'] = np.nan
```

* `df` – The DataFrame.
* `.loc[...]` – Access rows by index.
* `[5, 12, 25]` – Indexes of rows.
* `'Math_Score'` – Column name.
* `=` – Assignment.
* `np.nan` – Represents a missing value.

```python
df.loc[[3, 15], 'English_Score'] = np.nan
```

* Same as above but for different rows and the `English_Score` column.

---

### 7. **View Data**

```python
df
```

* Typing just `df` shows the DataFrame (in Jupyter Notebooks or similar).

---

### 8. **Check for Missing Values**

```python
print("\nMissing values before filling:\n", df.isnull().sum())
```

* `print(...)` – Displays output.
* `\n` – Line break.
* `df.isnull()` – Checks which values are missing (`True`/`False`).
* `.sum()` – Adds up the `True` values column-wise.

---

### 9. **Fill Missing Values**

```python
df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].median())
```

* `df[...]` – Access column.
* `.fillna(...)` – Replaces `NaN` with specified value.
* `.median()` – Gets the middle value.

```python
df['English_Score'] = df['English_Score'].fillna(df['English_Score'].median())
```

* Same logic, applied to the `English_Score` column.

---

### 10. **Check Again for Missing**

```python
print("\nMissing values after filling:\n", df.isnull().sum())
```

* Verifies that no missing values remain.

---

### 11. **Outlier Detection**

```python
z_scores = stats.zscore(df['Math_Score'])
```

* `z_scores` – Stores computed Z-scores.
* `stats.zscore(...)` – Calculates how many standard deviations each value is from the mean.

```python
outliers = np.where(np.abs(z_scores) > 3)[0]
```

* `np.abs(...)` – Takes the absolute value.
* `> 3` – Z-scores beyond 3 are considered outliers.
* `np.where(...)` – Returns indices of those outliers.
* `[0]` – Extracts the array from a tuple.

```python
print("\nOutlier indices (Math_Score):", outliers[0])
```

* Prints the index of the first outlier.

---

### 12. **Print and Replace Outlier**

```python
print("Outlier value in Math_Score:", df.loc[69, 'Math_Score'])
```

* Shows the actual outlier value at index 69.

```python
median_math = df['Math_Score'].median()
```

* Stores the median value.

```python
df.loc[outliers, 'Math_Score'] = median_math
```

* Replaces all detected outliers with the median.

```python
print("Outlier value after:", df.loc[69, 'Math_Score'])
```

* Verifies that the outlier value has been replaced.

---

### 13. **Plot Histogram (Original)**

```python
sns.histplot(df['Math_Score'], kde=True)
```

* `sns.histplot(...)` – Plots histogram.
* `kde=True` – Adds Kernel Density Estimation (smooth curve).

```python
plt.title("Original Math_Score Distribution")
```

* Sets the plot title.

```python
plt.show()
```

* Displays the plot.

---

### 14. **Log Transformation**

```python
df['Math_Score_Log'] = np.log(df['Math_Score'] + 1)
```

* Applies logarithmic transformation to reduce skew.
* Adds 1 to avoid log(0).
* Saves as a new column.

```python
sns.histplot(df['Math_Score_Log'], kde=True)
plt.title("Log-Transformed Math_Score Distribution")
plt.show()
```

* Plots the transformed scores with title.

---

Let me know if you'd like this turned into a graphic or flashcards for easier memorization.



