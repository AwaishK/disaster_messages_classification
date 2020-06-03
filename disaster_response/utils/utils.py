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
    return lemmatizer.lemmatize(word.lower().strip())

class WordCount(BaseEstimator, TransformerMixin):
    """This class is used to tansform the text to a number of words in text to be used as a feature.
    """
    def __init__(self, normalized=True):
        """
        params: 
            normalized: it is either True or False, if true normalized the number of words in text with number of words in data set
        """
        self.normalized = normalized
        
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        total = len(tokenize(" ".join(list(X))))
        count = lambda x: len(tokenize(x))/total if self.normalized else len(tokenize(x))
        X_tagged = pd.Series(X).apply(count)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    """This function is used to clean, tokenize and normalize the given string.
    params:
        text: string to be tokenized
    returns: tokenized words
    """
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    words = word_tokenize(text)
    words = [lemmatize(w) for w in words if w not in stop_set]
    return words
