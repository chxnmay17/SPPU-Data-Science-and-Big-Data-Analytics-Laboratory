#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("BostonHousing.csv")

# Display first few rows
print("Sample Data:\n", df.head())

# Features and Target
X = df.drop('medv', axis=1)  # MEDV is the target column (house price)
y = df['medv']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Comparison of actual vs predicted
comparison = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print("\nComparison of Actual and Predicted Prices:")
print(comparison.head(10))

# Plotting Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Home Prices')
plt.grid(True)
plt.show()


# In[ ]:

Here’s a **word-by-word and line-by-line explanation** of your Python code for linear regression using the Boston Housing dataset:

---

### 🔹 **Imports**

```python
import pandas as pd
```

* `import` – Loads an external module.
* `pandas` – A library for data manipulation.
* `as pd` – Aliases `pandas` to `pd` for shorter access.

```python
import matplotlib.pyplot as plt
```

* `matplotlib.pyplot` – A module for creating visualizations.
* `as plt` – Aliases it to `plt`.

```python
from sklearn.model_selection import train_test_split
```

* `from ... import` – Imports a specific function or class.
* `sklearn.model_selection` – Submodule for splitting data and tuning models.
* `train_test_split` – Function that splits the data into training and testing sets.

```python
from sklearn.linear_model import LinearRegression
```

* `sklearn.linear_model` – Submodule containing regression models.
* `LinearRegression` – A class that implements simple and multiple linear regression.

```python
from sklearn.metrics import mean_squared_error, r2_score
```

* `sklearn.metrics` – Contains functions to evaluate model performance.
* `mean_squared_error` – Measures average squared difference between predictions and true values.
* `r2_score` – Computes R² (coefficient of determination), a measure of model fit.

---

### 🔹 **Load dataset**

```python
df = pd.read_csv("BostonHousing.csv")
```

* `df` – Variable to hold the DataFrame.
* `pd` – Refers to pandas.
* `read_csv(...)` – Loads data from a CSV file.
* `"BostonHousing.csv"` – File name of the dataset.

---

### 🔹 **Preview the data**

```python
print("Sample Data:\n", df.head())
```

* `print(...)` – Outputs text and data to the screen.
* `"\n"` – New line.
* `df.head()` – Shows the first 5 rows of the DataFrame.

---

### 🔹 **Split features and target**

```python
X = df.drop('medv', axis=1)
```

* `X` – Input features (predictors).
* `df.drop(...)` – Removes a column.
* `'medv'` – Target variable (median home value).
* `axis=1` – Drop a **column** (axis 0 = rows, axis 1 = columns).

```python
y = df['medv']
```

* `y` – Target values (what we're trying to predict).
* `df['medv']` – Selects the `medv` column from the DataFrame.

---

### 🔹 **Split data into train and test sets**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

* `train_test_split(...)` – Splits data randomly.
* `X_train` – Training features.
* `X_test` – Testing features.
* `y_train` – Training target values.
* `y_test` – Testing target values.
* `test_size=0.2` – 20% data for testing.
* `random_state=42` – Ensures reproducibility.

---

### 🔹 **Create and train the model**

```python
model = LinearRegression()
```

* `model` – Variable holding the linear regression model.
* `LinearRegression()` – Instantiates the model.

```python
model.fit(X_train, y_train)
```

* `fit(...)` – Trains the model on training data (`X_train`, `y_train`).

---

### 🔹 **Make predictions**

```python
y_pred = model.predict(X_test)
```

* `predict(...)` – Uses the model to make predictions on unseen (test) data.

---

### 🔹 **Evaluate model**

```python
mse = mean_squared_error(y_test, y_pred)
```

* `mse` – Mean Squared Error value.
* `mean_squared_error(...)` – Compares actual and predicted values.

```python
r2 = r2_score(y_test, y_pred)
```

* `r2` – R-squared score (1 = perfect fit, 0 = no fit).

```python
print(f"\nMean Squared Error: {mse:.2f}")
```

* `f""` – f-string to embed variables inside a string.
* `{mse:.2f}` – Formats `mse` to 2 decimal places.
* `\n` – Line break.

```python
print(f"R² Score: {r2:.2f}")
```

* `{r2:.2f}` – Displays R² with 2 decimal places.

---

### 🔹 **Compare actual and predicted values**

```python
comparison = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
```

* `comparison` – New DataFrame showing side-by-side comparison.
* `pd.DataFrame(...)` – Creates a DataFrame from a dictionary.

```python
print("\nComparison of Actual and Predicted Prices:")
print(comparison.head(10))
```

* `comparison.head(10)` – Shows first 10 comparisons.

---

### 🔹 **Plot actual vs predicted**

```python
plt.figure(figsize=(8, 5))
```

* `figure(...)` – Creates a new plot with a specific size.

```python
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.6)
```

* `scatter(...)` – Creates a scatter plot.
* `y_test` – Actual prices.
* `y_pred` – Predicted prices.
* `color='blue'` – Dot color.
* `edgecolor='k'` – Black edges.
* `alpha=0.6` – 60% opacity.

```python
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
```

* `plot(...)` – Plots a diagonal reference line (perfect predictions).
* `'r--'` – Red dashed line.
* `lw=2` – Line width.

```python
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
```

* Labels for x and y axes.

```python
plt.title('Actual vs Predicted Home Prices')
```

* Title of the plot.

```python
plt.grid(True)
plt.show()
```

* `grid(True)` – Adds grid lines.
* `show()` – Displays the plot.

---

### 🔹 **Jupyter cell marker**

```python
# In[ ]:
```

* Used in Jupyter Notebook to indicate a code cell.
* Not executed.

---

Would you like this code visualized as a flow diagram?



