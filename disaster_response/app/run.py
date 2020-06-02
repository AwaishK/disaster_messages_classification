import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)


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
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///./disaster_response/data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("./disaster_response/models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == "__main__":
    main()
