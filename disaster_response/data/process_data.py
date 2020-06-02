import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """This function is used to load the data from csv files and return merged dataframe.
    params:
        messages_filepath: contains the path to csv file for messages data.
        categories_filepath: contains the path to csv file for category data.
    returns: merged dataframe.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """This function is used to clean the dataframe.
    params:
        df: df to clean.
    returns cleaned dataframe.
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    row = categories.loc[0]
    category_colnames = [r.split('-')[0] for r in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """This function is used to save to data from dataframe to sqlite database.
    params:
        df: dataframe containing the data
        database_filename: filename for database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()