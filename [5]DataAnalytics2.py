#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Social_Network_Ads.csv")

# Show column names
print(df.columns)

# Show first few rows
print(df.head())


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features and Target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[3]:


from sklearn.linear_model import LogisticRegression

# Train model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)


# In[4]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


# In[5]:


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:

EXPLAIN****************************************************************************************************************************
Absolutely! Here's a **word-by-word and line-by-line explanation** of your full code with **simple definitions** for each term.

---

## 🔹 Full Code Explanation (Every Word)

### ✅ Line 1–3:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

* `import`: keyword to bring external code into your program.
* `pandas`: a Python library for handling tables and spreadsheets.
* `as pd`: you can now type `pd` instead of `pandas`.
* `matplotlib.pyplot`: used for plotting graphs.
* `as plt`: lets you use `plt` instead of the full name.
* `seaborn`: another library for prettier charts and statistics.
* `as sns`: now you can type `sns` instead of `seaborn`.

---

### ✅ Load dataset:

```python
df = pd.read_csv("Social_Network_Ads.csv")
```

* `df`: short for DataFrame, a table (rows + columns).
* `pd.read_csv(...)`: reads a `.csv` file and loads it into a table.

---

### ✅ Show columns and rows:

```python
print(df.columns)
print(df.head())
```

* `print(...)`: display something.
* `df.columns`: shows the list of column names.
* `df.head()`: shows the first 5 rows of the data.

---

### ✅ Importing tools for preprocessing:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

* `from ... import ...`: brings only specific tools from a library.
* `sklearn`: a machine learning library.
* `model_selection`: handles splitting data for training/testing.
* `train_test_split`: splits data into training and testing parts.
* `preprocessing`: for preparing data.
* `StandardScaler`: scales (normalizes) numeric data to standard format.

---

### ✅ Set features and target:

```python
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
```

* `X`: input features (Age and Salary).
* `df[['...']]`: selects columns from the DataFrame.
* `y`: target/output (what you want to predict).
* `df['Purchased']`: selects the "Purchased" column (yes/no).

---

### ✅ Split the data:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

* `X_train`: part of X for training.
* `X_test`: part of X for testing.
* `y_train`: part of y for training.
* `y_test`: part of y for testing.
* `test_size=0.25`: 25% of the data is for testing.
* `random_state=0`: ensures same result every time.

---

### ✅ Normalize (scale) the data:

```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

* `sc`: scaler object.
* `StandardScaler()`: prepares scaler.
* `fit_transform`: finds mean/std and transforms training data.
* `transform`: applies same scaling to test data.

---

### ✅ Import and train logistic regression:

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
```

* `LogisticRegression`: a model to predict binary outcomes (yes/no).
* `classifier`: name for the model.
* `fit`: trains the model using the training data.

---

### ✅ Make predictions:

```python
y_pred = classifier.predict(X_test)
```

* `y_pred`: predicted output.
* `predict(...)`: use model to guess output from test data.

---

### ✅ Evaluation metrics:

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
```

* `metrics`: tools to measure model performance.
* `confusion_matrix`: shows correct/wrong predictions.
* `accuracy_score`: % of correct predictions.
* `precision_score`: how many positive predictions were correct.
* `recall_score`: how many actual positives were caught.

---

### ✅ Confusion Matrix Breakdown:

```python
cm = confusion_matrix(y_test, y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
```

* `cm`: 2x2 table showing model prediction results.
* `TP`: true positives (predicted yes, was yes).
* `TN`: true negatives (predicted no, was no).
* `FP`: false positives (predicted yes, was no).
* `FN`: false negatives (predicted no, was yes).

---

### ✅ Calculate metrics:

```python
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
```

* `accuracy`: correct predictions ÷ total predictions.
* `error_rate`: incorrect predictions ÷ total.
* `precision`: correct yes ÷ total predicted yes.
* `recall`: correct yes ÷ total actual yes.

---

### ✅ Print results:

```python
print("Confusion Matrix:\n", cm)
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
```

* `print(...)`: show output.
* `f"..."`: formatted string.
* `:.2f`: round number to 2 decimal places.

---

### ✅ Plot Confusion Matrix:

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

* `sns.heatmap`: draws the confusion matrix as a grid.
* `annot=True`: shows numbers in the grid.
* `fmt='d'`: format as digits (integers).
* `cmap='Blues'`: use blue shades.
* `plt.xlabel`: label x-axis.
* `plt.ylabel`: label y-axis.
* `plt.title`: title of the graph.
* `plt.show()`: display the plot.

---

Would you like this as a **PDF or visual diagram** for easier reference?



