#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd

# Load the built-in iris dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  # This is already encoded as 0, 1, 2

print(df.head())


# In[6]:


# Encode the categorical target variable
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[7]:


from sklearn.naive_bayes import GaussianNB

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# In[8]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Assuming binary classification for TP/FP/TN/FN calculation
# (but since Iris has 3 classes, we compute per-class if needed)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='macro')  # or 'weighted'
recall = recall_score(y_test, y_pred, average='macro')

print("Confusion Matrix:\n", cm)
print(f"Accuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:

EXPLAIN***********************************************************************************************************************
  Here is a **word-by-word and line-by-line explanation** of the complete code using the **Iris dataset with Naïve Bayes classification**. Each term is explained in simple words.

---

## ✅ Step-by-step Explanation

### 🔹 Import Required Modules

```python
from sklearn.datasets import load_iris
```

* `from`: keyword to import specific items.
* `sklearn.datasets`: module containing sample datasets.
* `load_iris`: a function to load the Iris dataset.

```python
from sklearn.preprocessing import LabelEncoder
```

* `sklearn.preprocessing`: module for data preprocessing.
* `LabelEncoder`: converts labels (like strings) into numbers.

```python
from sklearn.model_selection import train_test_split
```

* `model_selection`: module to split data into training and test sets.
* `train_test_split`: function to divide data into training and testing parts.

```python
import pandas as pd
```

* `import`: bring in a package.
* `pandas`: a library for handling tabular data.
* `as pd`: allows you to use `pd` as short for `pandas`.

---

### 🔹 Load and Convert the Dataset

```python
iris = load_iris()
```

* `iris`: now holds the data returned by `load_iris()` — a dictionary-like object with keys like `data`, `target`, and `feature_names`.

```python
df = pd.DataFrame(iris.data, columns=iris.feature_names)
```

* `pd.DataFrame(...)`: create a table (DataFrame).
* `iris.data`: the features (measurements).
* `columns=iris.feature_names`: set column names like 'sepal length'.

```python
df['species'] = iris.target
```

* `df['species']`: creates a new column named 'species'.
* `iris.target`: values 0, 1, or 2 (which species it is).

```python
print(df.head())
```

* `df.head()`: displays the first 5 rows of the table.

---

### 🔹 Encode Target Variable

```python
le = LabelEncoder()
```

* `le`: variable that stores the label encoder object.

```python
df['species'] = le.fit_transform(df['species'])
```

* `le.fit_transform(...)`: fits the encoder and transforms values (0, 1, 2) — though here it's already numeric, so no change.

---

### 🔹 Select Features and Target

```python
X = df.drop('species', axis=1)
```

* `df.drop(...)`: removes a column.
* `'species'`: column name to remove.
* `axis=1`: tells it to remove a column (not a row).
* `X`: stores input features (4 columns).

```python
y = df['species']
```

* `y`: stores the target variable (species number).

---

### 🔹 Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* `X_train`: 70% of input data for training.
* `X_test`: 30% for testing.
* `y_train`: corresponding labels for training.
* `y_test`: labels for testing.
* `test_size=0.3`: 30% of data for testing.
* `random_state=42`: fixes the randomness for same results every time.

---

### 🔹 Train Naïve Bayes Model

```python
from sklearn.naive_bayes import GaussianNB
```

* `naive_bayes`: module for Naïve Bayes models.
* `GaussianNB`: type of Naïve Bayes that works for numerical features assuming Gaussian (normal) distribution.

```python
model = GaussianNB()
```

* `model`: stores the classifier.

```python
model.fit(X_train, y_train)
```

* `fit(...)`: trains the model with training data.

---

### 🔹 Make Predictions

```python
y_pred = model.predict(X_test)
```

* `predict(...)`: uses the model to predict on test data.
* `y_pred`: predicted species values.

---

### 🔹 Evaluate Model

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
```

* `metrics`: module for checking performance.
* `confusion_matrix`: compares actual and predicted labels.
* `accuracy_score`: % of correct predictions.
* `precision_score`: how many predicted species are correct.
* `recall_score`: how many actual species were correctly found.

---

### 🔹 Compute Metrics

```python
cm = confusion_matrix(y_test, y_pred)
```

* `cm`: matrix comparing actual vs predicted classes.

```python
accuracy = accuracy_score(y_test, y_pred)
```

* `accuracy`: correct ÷ total.

```python
error_rate = 1 - accuracy
```

* `error_rate`: incorrect ÷ total.

```python
precision = precision_score(y_test, y_pred, average='macro')
```

* `precision`: average precision across classes (macro means equal weight).

```python
recall = recall_score(y_test, y_pred, average='macro')
```

* `recall`: average recall across classes.

---

### 🔹 Print Results

```python
print("Confusion Matrix:\n", cm)
print(f"Accuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
```

* `print(...)`: display each metric.
* `:.2f`: show numbers with 2 decimal places.

---

### 🔹 Visualize Confusion Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

* `seaborn`: for attractive charts.
* `matplotlib.pyplot`: base library for plots.

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
```

* `sns.heatmap(...)`: draw grid image of matrix.
* `annot=True`: show numbers inside boxes.
* `fmt='d'`: format numbers as integers.
* `cmap='Greens'`: use green shades.

```python
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

* `plt.title(...)`: chart title.
* `plt.xlabel(...)`: label below chart.
* `plt.ylabel(...)`: label on side.
* `plt.show()`: display chart.

---

Would you like a printable PDF of this full explanation?



