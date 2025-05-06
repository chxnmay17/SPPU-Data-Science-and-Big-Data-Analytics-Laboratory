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
EXPLAON*************************************************************************************************************************************************************************8
Here is a **word-by-word and line-by-line explanation** of your code for analyzing the Titanic dataset using Seaborn and Matplotlib:

---

## 🔹 **Importing Libraries**

```python
import seaborn as sns
```

* `import`: Python keyword to bring a module/library into your code.
* `seaborn`: A Python data visualization library built on top of `matplotlib`.
* `sns`: Alias used to refer to `seaborn` for convenience.

```python
import matplotlib.pyplot as plt
```

* `matplotlib`: Core plotting library.
* `pyplot`: Sub-module for creating plots.
* `plt`: Alias for `pyplot`.

---

## 🔹 **Load Titanic Dataset**

```python
titanic = sns.load_dataset('titanic')
```

* `sns.load_dataset(...)`: Loads built-in datasets from Seaborn.
* `'titanic'`: Name of the dataset (about passengers on the Titanic).
* `titanic`: Variable that stores the dataset as a **DataFrame** (a table).

---

## 🔹 **Display First Few Rows**

```python
print(titanic.head())
```

* `print(...)`: Outputs to console.
* `titanic.head()`: Shows the first 5 rows of the dataset.

---

## 🔹 **Survival Count by Gender**

```python
sns.countplot(x='sex', hue='survived', data=titanic)
```

* `sns.countplot(...)`: Creates a bar chart showing counts of data categories.
* `x='sex'`: X-axis is gender (male/female).
* `hue='survived'`: Different colors for survived (`1`) and not survived (`0`).
* `data=titanic`: Use Titanic dataset.

```python
plt.title('Survival by Gender')
```

* `plt.title(...)`: Sets title of the plot.

```python
plt.show()
```

* Displays the plot on screen.

---

## 🔹 **Survival Count by Passenger Class**

```python
sns.countplot(x='class', hue='survived', data=titanic)
```

* Similar to the previous plot.
* `x='class'`: X-axis now shows ticket class (`First`, `Second`, `Third`).

```python
plt.title('Survival by Passenger Class')
plt.show()
```

* Adds title and displays the chart.

---

## 🔹 **Age Distribution**

```python
sns.histplot(titanic['age'].dropna(), kde=True)
```

* `sns.histplot(...)`: Creates a histogram (frequency distribution).
* `titanic['age']`: Selects the `age` column.
* `.dropna()`: Removes missing values (NaN).
* `kde=True`: Adds a smooth curve (Kernel Density Estimate) over histogram.

```python
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
```

* Labels and displays the plot.

---

## 🔹 **Fare Distribution**

```python
sns.histplot(titanic['fare'], kde=True, bins=30)
```

* Histogram of ticket prices (`fare`).
* `bins=30`: Divide the fare range into 30 intervals/bars.
* `kde=True`: Adds density curve.

```python
plt.title('Ticket Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.show()
```

* Title and axis labels for the fare distribution plot.

---

## 📌 Summary of Key Terms:

| Word / Function            | Meaning                                         |
| -------------------------- | ----------------------------------------------- |
| `import`                   | Load a module into your program.                |
| `seaborn as sns`           | Visualization library for statistical graphics. |
| `matplotlib.pyplot as plt` | Core plotting functions.                        |
| `load_dataset('titanic')`  | Loads Titanic dataset from seaborn.             |
| `head()`                   | Shows first 5 rows of DataFrame.                |
| `countplot(...)`           | Bar chart showing counts by category.           |
| `histplot(...)`            | Histogram for numerical data.                   |
| `dropna()`                 | Removes missing/null values.                    |
| `kde=True`                 | Adds a smooth density curve.                    |
| `xlabel`, `ylabel`         | Labels for X and Y axes.                        |
| `title()`                  | Sets the title of the plot.                     |
| `show()`                   | Displays the plot.                              |

---

Would you like explanations of each column in the Titanic dataset as well?




