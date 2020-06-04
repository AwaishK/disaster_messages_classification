import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import pandas as pd
from functools import lru_cache

from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin

# populate the stop words 
stop_set = set(stopwords.words())
# initilize lemmatizer
lemmatizer = WordNetLemmatizer()

# trying to cache the lematizer.lemmatize function to speedup the transformer step in pipeline
@lru_cache(maxsize=10000)
def lemmatize(word):
    """Gets the string and normalize and lemmatize it.

    Parameters
    ----------
    word : string
        a string to be lemmatized
    Returns
    -------
    string
        lemmatized and normalized string
    """
    return lemmatizer.lemmatize(word.lower().strip())

class WordCount(BaseEstimator, TransformerMixin):
    """
    A class used to tansform the text to a number of words in text to be used as a feature.
    ...

    Attributes
    ----------
    normalized : bool, optional
        a flag to normalize the word count in string with word count in data set

    Methods
    -------
    fit(self, x, y=None)
        fits the data set
    transform(self, X)
        transform the given data set and returns transformed feature set    
    """
    def __init__(self, normalized=True):
        """
        Parameters
        ----------
        normalized : bool, optional
            a flag to normalize the word count in string with word count in data set
        """
        self.normalized = normalized
        
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """Gets the data set and tranform it. Transformation involved word count in string.

        Parameters
        ----------
        X : Series
            The series of string text
        Returns
        -------
        DataFrame
            a dataframe containing the count of words
        """
        total = len(tokenize(" ".join(list(X))))
        count = lambda x: len(tokenize(x))/total if self.normalized else len(tokenize(x))
        X_tagged = pd.Series(X).apply(count)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    """Gets the text and  clean, tokenize and normalize the given text.

    Parameters
    ----------
    text : string
        string to be tokenized
    Returns
    -------
    list
        list of tokens
    """
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    words = word_tokenize(text)
    words = [lemmatize(w) for w in words if w not in stop_set]
    return words
