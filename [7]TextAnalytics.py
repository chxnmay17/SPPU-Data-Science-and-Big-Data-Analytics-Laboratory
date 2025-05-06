#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data (run once)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[3]:


doc = "Text analytics is the process of extracting meaningful information from unstructured text."
print("Original Document:\n", doc)


# In[4]:


tokens = word_tokenize(doc)
print("\nTokenized Words:\n", tokens)


# In[6]:


nltk.download('averaged_perceptron_tagger_eng')
pos_tags = pos_tag(tokens)
print("\nPOS Tags:\n", pos_tags)


# In[7]:


stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words and w not in string.punctuation]
print("\nAfter Stop Words Removal:\n", filtered_tokens)


# In[8]:


stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_tokens]
print("\nAfter Stemming:\n", stemmed)


# In[9]:


lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word.lower()) for word in filtered_tokens]
print("\nAfter Lemmatization:\n", lemmatized)


# In[10]:


documents = [
    "Text analytics extracts meaningful information from text.",
    "Natural language processing is used in text analytics.",
    "Machine learning improves text classification."
]


# In[11]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Show TF-IDF values
print("\nTF-IDF Matrix:\n")
print(X.toarray())

# Show feature names (terms)
print("\nFeature Names:\n", vectorizer.get_feature_names_out())


# In[ ]:

Here is a **word-by-word and line-by-line explanation** of your complete **Python NLP (Natural Language Processing)** script using `nltk` and `scikit-learn`:

---

## ✅ **Header Comments**

```python
#!/usr/bin/env python
```

* `#!`: Shebang line — tells the system which interpreter to use.
* `/usr/bin/env python`: uses the default Python interpreter in the system environment.

```python
# coding: utf-8
```

* Declares the encoding of the script.
* `utf-8`: supports all characters (including emojis, foreign languages, etc.).

---

## 🔹 **Import Libraries**

```python
import nltk
```

* `nltk`: Natural Language Toolkit — a library for text processing.

```python
import string
```

* `string`: standard Python module for string operations (e.g., punctuation list).

```python
import numpy as np
```

* `numpy`: library for numerical arrays and matrix operations.
* `as np`: lets you refer to it as `np`.

```python
from nltk.corpus import stopwords
```

* `corpus`: collection of linguistic data.
* `stopwords`: common words like *"the", "is", "and"* — often removed.

```python
from nltk.tokenize import word_tokenize
```

* `tokenize`: breaks text into words or sentences.
* `word_tokenize`: splits text into individual words.

```python
from nltk import pos_tag
```

* `pos_tag`: assigns part-of-speech (POS) tags like noun, verb, adjective to each word.

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
```

* `stem`: reduce words to base form (e.g., "running" → "run").
* `PorterStemmer`: a rule-based stemmer.
* `WordNetLemmatizer`: reduces words to meaningful root using vocabulary.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

* `feature_extraction.text`: for text to number conversion.
* `TfidfVectorizer`: converts documents into a matrix of **TF-IDF** scores.

---

## 🔹 **Download Required NLTK Resources**

```python
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

* Downloads models/datasets needed by NLTK:

  * `'punkt_tab'`: sentence/word tokenization (but likely a typo; should be `'punkt'`).
  * `'averaged_perceptron_tagger'`: for POS tagging.
  * `'stopwords'`: list of stopwords.
  * `'wordnet'`: database used for lemmatization.
  * `'omw-1.4'`: multilingual WordNet extension.

---

## 🔹 **Original Text**

```python
doc = "Text analytics is the process of extracting meaningful information from unstructured text."
```

* `doc`: a single string holding a sentence.

```python
print("Original Document:\n", doc)
```

* Shows the original sentence.

---

## 🔹 **Tokenization**

```python
tokens = word_tokenize(doc)
```

* Splits sentence into words: e.g., `"Text"` → `"Text"`, `"analytics"` → `"analytics"`, etc.

```python
print("\nTokenized Words:\n", tokens)
```

* Prints list of tokenized words.

---

## 🔹 **Part-of-Speech Tagging**

```python
nltk.download('averaged_perceptron_tagger_eng')
```

* Downloads English POS tagger (typo again — `'averaged_perceptron_tagger'` is sufficient).

```python
pos_tags = pos_tag(tokens)
```

* `pos_tag(...)`: assigns grammatical roles to each word.

  * Example: `("Text", "NN")`, `("is", "VBZ")`

```python
print("\nPOS Tags:\n", pos_tags)
```

* Shows each word with its POS tag.

---

## 🔹 **Stop Word Removal**

```python
stop_words = set(stopwords.words('english'))
```

* Loads English stop words into a `set` (faster search).

```python
filtered_tokens = [w for w in tokens if w.lower() not in stop_words and w not in string.punctuation]
```

* List comprehension that:

  * Converts `w` to lowercase.
  * Removes if `w` is a stopword or punctuation (`.`, `,`, etc).

```python
print("\nAfter Stop Words Removal:\n", filtered_tokens)
```

* Prints the result after filtering.

---

## 🔹 **Stemming**

```python
stemmer = PorterStemmer()
```

* Creates a stemmer object.

```python
stemmed = [stemmer.stem(word) for word in filtered_tokens]
```

* Applies stemming to each word in the filtered list.

```python
print("\nAfter Stemming:\n", stemmed)
```

* Shows stemmed words (e.g., `"extracting"` → `"extract"`).

---

## 🔹 **Lemmatization**

```python
lemmatizer = WordNetLemmatizer()
```

* Creates a lemmatizer object.

```python
lemmatized = [lemmatizer.lemmatize(word.lower()) for word in filtered_tokens]
```

* Lemmatizes each word to its base form using vocabulary and grammar rules.

```python
print("\nAfter Lemmatization:\n", lemmatized)
```

* Shows lemmatized results (e.g., `"extracting"` → `"extract"`).

---

## 🔹 **TF-IDF Vectorization**

```python
documents = [
    "Text analytics extracts meaningful information from text.",
    "Natural language processing is used in text analytics.",
    "Machine learning improves text classification."
]
```

* A list of three short documents.

```python
vectorizer = TfidfVectorizer()
```

* Creates a TF-IDF object.

```python
X = vectorizer.fit_transform(documents)
```

* Learns vocabulary and computes TF-IDF scores.
* Returns a **sparse matrix**.

```python
print("\nTF-IDF Matrix:\n")
print(X.toarray())
```

* Converts sparse matrix to full array and prints the TF-IDF values for each word in each document.

```python
print("\nFeature Names:\n", vectorizer.get_feature_names_out())
```

* Prints the vocabulary/terms in order of columns in the matrix.

---

Would you like me to highlight the **differences between stemming and lemmatization** or **explain TF-IDF** in detail next?



