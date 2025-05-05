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




