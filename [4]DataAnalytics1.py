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
EXPLAIN***************************************************************************************************************************
Thank you for your feedback! Let's now go step by step and **explain every word and line in depth**, assuming you're a **complete beginner**. I'll break it down clearly:

---

## ✅ Full Code Line-by-Line and Word-by-Word Explanation

### 🔹 1. Import Required Libraries

```python
import pandas as pd
```

* `import` – This keyword brings an external module into your Python program.
* `pandas` – A library that helps with data handling, like reading CSV files and working with tables.
* `as pd` – This gives pandas a short name (`pd`), so we don’t have to type `pandas` each time.

---

```python
import matplotlib.pyplot as plt
```

* `matplotlib` – A library for creating graphs and charts.
* `.pyplot` – A sub-module of matplotlib, used specifically for plotting.
* `as plt` – Again, shortens the name so you can just type `plt` to access pyplot functions.

---

```python
from sklearn.model_selection import train_test_split
```

* `from ... import` – This format brings only a specific part (not the whole module).
* `sklearn` – A machine learning library.
* `model_selection` – A sub-module of sklearn for splitting datasets.
* `train_test_split` – A function that divides your data into training and testing sets.

---

```python
from sklearn.linear_model import LinearRegression
```

* `linear_model` – A sub-module inside `sklearn` for linear-type models.
* `LinearRegression` – A class that creates a model to fit a straight line to your data.

---

```python
from sklearn.metrics import mean_squared_error, r2_score
```

* `metrics` – A sub-module for checking how good your model is.
* `mean_squared_error` – Measures how far off your predictions are from actual values.
* `r2_score` – Measures how well your model explains the variation in data (closer to 1 is better).

---

## 🔹 2. Load the Data

```python
df = pd.read_csv("BostonHousing.csv")
```

* `df` – Short for DataFrame, a table-like data structure from pandas.
* `pd` – That’s pandas, because we used `import pandas as pd`.
* `.read_csv(...)` – A function to load a CSV file (Comma-Separated Values).
* `"BostonHousing.csv"` – The file name to load (should be in the same folder).

---

## 🔹 3. Display First Few Rows

```python
print("Sample Data:\n", df.head())
```

* `print(...)` – Shows output in the terminal.
* `"Sample Data:\n"` – A message. `\n` adds a line break.
* `df.head()` – Shows the first 5 rows of the dataset.

---

## 🔹 4. Separate Features and Target Column

```python
X = df.drop('medv', axis=1)
```

* `X` – Variable to store **features** (independent variables).
* `df.drop(...)` – Drops the `'medv'` column from the DataFrame.
* `'medv'` – This is the target column (the price of the house).
* `axis=1` – Means "drop a column" (if it were `axis=0`, it would drop a row).

---

```python
y = df['medv']
```

* `y` – Variable to store **target values** (dependent variable).
* `df['medv']` – Selects the `'medv'` column from the DataFrame.

---

## 🔹 5. Split into Training and Testing Sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

* `X_train` – 80% of the features for training the model.
* `X_test` – 20% of the features for testing the model.
* `y_train` – 80% of the target values for training.
* `y_test` – 20% of the target values for testing.
* `train_test_split(...)` – Function that splits the data randomly.
* `test_size=0.2` – 20% of the data goes to testing.
* `random_state=42` – A fixed seed to get the same result every time.

---

## 🔹 6. Create and Train the Linear Regression Model

```python
model = LinearRegression()
```

* `model` – A variable to store your machine learning model.
* `LinearRegression()` – Creates a new model object that can learn a straight-line relationship.

---

```python
model.fit(X_train, y_train)
```

* `fit(...)` – This function trains the model on the training data.
* `X_train` – The input data.
* `y_train` – The correct output data (house prices).

---

## 🔹 7. Predict Using the Model

```python
y_pred = model.predict(X_test)
```

* `y_pred` – Variable that stores predictions made by the model.
* `model.predict(...)` – This asks the model to make predictions on the test data (`X_test`).

---

## 🔹 8. Evaluate the Model

```python
mse = mean_squared_error(y_test, y_pred)
```

* `mse` – Variable to store the **mean squared error**.
* `mean_squared_error(...)` – Tells how far predictions are from real values (lower is better).
* `y_test` – Actual prices.
* `y_pred` – Predicted prices.

---

```python
r2 = r2_score(y_test, y_pred)
```

* `r2` – Variable to store **R-squared score**.
* `r2_score(...)` – Measures how well predictions match actual data (1.0 is perfect).

---

```python
print(f"\nMean Squared Error: {mse:.2f}")
```

* `f"...{value}..."` – f-string lets you put variables inside strings.
* `{mse:.2f}` – Format MSE to 2 decimal places.
* `\n` – Adds a blank line before printing.

---

```python
print(f"R² Score: {r2:.2f}")
```

* `R² Score` – R-squared value is printed here to judge accuracy.

---

## 🔹 9. Compare Actual and Predicted Values

```python
comparison = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
```

* `comparison` – A new table showing both real and predicted prices.
* `pd.DataFrame({...})` – Makes a table from a dictionary of two lists.

---

```python
print("\nComparison of Actual and Predicted Prices:")
print(comparison.head(10))
```

* Shows the first 10 values in the comparison.

---

## 🔹 10. Plotting the Results

```python
plt.figure(figsize=(8, 5))
```

* Creates a new blank figure that is 8 inches wide and 5 inches tall.

---

```python
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.6)
```

* `scatter(...)` – Makes a scatter plot (dots).
* `y_test` – Actual house prices (X-axis).
* `y_pred` – Predicted prices (Y-axis).
* `color='blue'` – Dots are blue.
* `edgecolor='k'` – Border of dots is black.
* `alpha=0.6` – Dots are slightly transparent.

---

```python
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
```

* Draws a red dashed line from the minimum to maximum price.
* This line shows perfect prediction: if your model were 100% accurate, all points would be on this line.

---

```python
plt.xlabel('Actual Price')
```

* Adds a label to the X-axis.

```python
plt.ylabel('Predicted Price')
```

* Adds a label to the Y-axis.

```python
plt.title('Actual vs Predicted Home Prices')
```

* Adds a title to the chart.

```python
plt.grid(True)
```

* Shows grid lines in the background.

```python
plt.show()
```

* Displays the plot.

---

### 🔹 Final Marker (only used in notebooks)

```python
# In[ ]:
```

* This line has no effect. It’s a placeholder from Jupyter Notebook.

---

Would you like me to give this as a diagram, PDF, or slides to help even more?




