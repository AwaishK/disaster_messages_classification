from sklearn.base import BaseEstimator, TransformerMixin
from models.train_classifier import WordCount, tokenize
from webapp import app

app.run(host='0.0.0.0', port=3001, debug=True)
