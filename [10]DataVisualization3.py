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
EXPLAIN*************************************************************************************************************************************
Here’s a detailed **word-by-word, line-by-line explanation** of your Iris dataset code:

---

## ✅ Step 1: Import Required Libraries

```python
import pandas as pd
```

* `import` → brings in external code (a package or module).
* `pandas` → a powerful data analysis library.
* `as pd` → lets you use `pd` instead of typing `pandas` each time.

```python
import matplotlib.pyplot as plt
```

* `matplotlib.pyplot` → a sub-library used for plotting graphs like histograms and boxplots.
* `as plt` → gives it the short alias `plt`.

```python
import seaborn as sns
```

* `seaborn` → another plotting library built on top of `matplotlib`, makes beautiful and informative statistical plots.
* `as sns` → short alias to use it more easily.

```python
from sklearn.datasets import load_iris
```

* `from` → used to import a specific part of a library.
* `sklearn.datasets` → a module in scikit-learn containing datasets.
* `load_iris` → a function that loads the Iris dataset.

---

## 🌸 Step 2: Load the Iris Dataset

```python
iris_sklearn = load_iris()
```

* `load_iris()` → loads the Iris dataset.
* `iris_sklearn` → stores the dataset, which is like a dictionary with data, target, feature names, etc.

```python
iris = pd.DataFrame(data=iris_sklearn.data, columns=iris_sklearn.feature_names)
```

* `pd.DataFrame()` → creates a table (DataFrame) from the data.
* `data=...` → uses the feature values (lengths and widths).
* `columns=...` → sets the column names using `feature_names`.

```python
iris['species'] = iris_sklearn.target
```

* Adds a new column called `species` to the DataFrame with integer values (0, 1, 2) representing species.

```python
iris['species'] = iris['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
```

* `.map()` → converts those numbers to actual species names for better readability.

---

## 🧾 Step 3: List Features and Their Types

```python
print("\n1. Features and their types:\n")
print(iris.dtypes)
```

* `dtypes` → tells you the data type of each column (e.g., float64, object).

```python
print("\nFeature Types Summary:")
print(iris.info())
```

* `.info()` → shows more detailed information: number of non-null entries, column names, and types.

---

## 📊 Step 4: Plot Histograms

```python
print("\n2. Histograms for each feature:")
```

* Displays a label for the next section.

```python
iris.iloc[:, :-1].hist(bins=15, figsize=(10, 6), layout=(2, 2), color='skyblue', edgecolor='black')
```

* `iloc[:, :-1]` → selects all columns except the last (`species`).
* `.hist()` → makes histograms for all numeric features.
* `bins=15` → breaks values into 15 bins.
* `figsize=(10, 6)` → sets the plot size.
* `layout=(2, 2)` → arranges the 4 plots in a 2x2 grid.
* `color='skyblue'`, `edgecolor='black'` → visual customization.

```python
plt.suptitle("Histograms of Iris Features", fontsize=14)
```

* Adds a big title for all subplots.

```python
plt.tight_layout()
plt.show()
```

* `tight_layout()` → avoids overlapping.
* `show()` → displays the plot.

---

## 📦 Step 5: Boxplots for Each Feature by Species

```python
print("\n3. Boxplots for each feature:")
```

```python
plt.figure(figsize=(12, 8))
```

* Starts a new figure with a larger size.

```python
for i, column in enumerate(iris.columns[:-1]):
```

* `for` loop to create one boxplot for each feature.
* `enumerate()` → gives both index (`i`) and column name (`column`).

```python
    plt.subplot(2, 2, i + 1)
```

* Places each plot in a 2x2 grid layout.

```python
    sns.boxplot(data=iris, x='species', y=column)
```

* `sns.boxplot()` → creates a boxplot.
* `x='species'` → categorizes by species.
* `y=column` → plots the feature (e.g., sepal length).

```python
    plt.title(f'Boxplot of {column}')
```

* Sets title for each boxplot.

```python
plt.tight_layout()
plt.show()
```

* Lays out plots nicely and displays them.

---

## 🚨 Step 6: Detect Outliers Using IQR Method

```python
print("\n4. Inference:")
```

```python
for col in iris.columns[:-1]:
```

* Loops through all numeric features (excluding `species`).

```python
    Q1 = iris[col].quantile(0.25)
    Q3 = iris[col].quantile(0.75)
    IQR = Q3 - Q1
```

* Computes the interquartile range (IQR = Q3 − Q1), used to find outliers.

```python
    outliers = iris[(iris[col] < (Q1 - 1.5 * IQR)) | (iris[col] > (Q3 + 1.5 * IQR))]
```

* Finds rows where the value is **less than Q1 - 1.5×IQR** or **more than Q3 + 1.5×IQR** → common rule to detect outliers.

```python
    print(f"{col}: {len(outliers)} outlier(s) found.")
```

* Prints the number of outliers for each column.

---

Would you like a PDF version of this annotated explanation or a simplified version for quick memorization?
