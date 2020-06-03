import sys, os
sys.path.append(os.path.join('models', '..'))
from utils.utils import WordCount, tokenize

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import sys
import pandas as pd
import re
from sklearn.externals import joblib
from sqlalchemy import create_engine
from functools import lru_cache

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """this function is used to load the data from database and split it into features and target.
    params:
        database_filepath: path of database file with extention .db
    returns: features as X, target as Y and classification classes as category_names
    """
   
    # create the engine and read data
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', engine) 

    # features
    X = df["message"]

    # figure out classification classes
    category_names = list(set(df.columns) - {"message", "id", "original", "genre", "child_alone"})
    Y = df[category_names]
    return X, Y, category_names


def build_model():
    """This function used to define the pipeline, parameters and build the multi-class classification model.
    returns: model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('word_count', WordCount())
        ])),
        ('clf', MultiOutputClassifier(LinearSVC()))
    ])

    parameters = {
            'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4)),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """This function is used to evaluate the model using recall, precision and f1-score using classification_report from sklearn
    params:
        model: trained classification model.
        X_test: features to test the model on.
        Y_test: target classes for test data.
        category_names: classes names
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names) 
    average = 0
    for col in category_names:
        print(col)
        print(classification_report(Y_test[col], y_pred_df[col]))
        accuracy = (y_pred_df[col] == Y_test[col]).mean()
        average += accuracy
        print(accuracy)
    print(average/ len(category_names))


def save_model(model, model_filepath):
    """save the model to a pickle file
    params:
        model: trained classification model.
        model_filepath: file path to store the model 
    """
    joblib.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        Y_train = Y_train.reset_index()
        Y_train.drop('index', axis=1, inplace=True)

        Y_test = Y_test.reset_index()
        Y_test.drop('index', axis=1, inplace=True)
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()