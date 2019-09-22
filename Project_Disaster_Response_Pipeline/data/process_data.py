import sys
import pandas as pd
import sqlalchemy as db
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    The data load with all the input parameters

    Args: 
    
    messages_filepath (str): The input file with the messages information - CSV file
    categories_filepath (str) : The input file with the categories information - CSV file

    Returns:
    Pandas data frame: A final merged dataframe obtained by merging the message file and categories file   

    """
    # load messages dataset
    messages = pd.read_csv(r'disaster_messages.csv')
    # load categories dataset
    categories = pd.read_csv(r'disaster_categories.csv')
    # merge datasets
    df = pd.merge(messages, categories, on='id', how='left')

    return df


def clean_data(df):
    """
    Cleans the merged dataframe for use by ML model
    
    Args:
    df (pandas_dataframe): Merged dataframe returned from the above load_data() function
    
    Returns:
    df (pandas_dataframe): Cleaned data to be used by ML model
    
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].astype(str).str.split(';', expand=True)   
    
    # select the first row of the categories dataframe
    row = pd.DataFrame()
    row = categories.iloc[0].to_frame()
    
    row.columns = ['col_name']
    for col in row.columns:
        row[col] = row[col].apply(lambda x: x[:-2])
        
    category_colnames = list(row['col_name'].values)
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1].astype(int)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    """
    Saves the cleaned data to an SQL database
    
    Args:
    
    df(pandas_dataframe): Cleaned data returned from clean_data() function
    
    database_file_name(str): File path of SQL Database into which the cleaned data is to be saved
    
    Returns:
    
    None
    
    """  
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    db_file_name = database_filename.split("/")[-1] # extract file name from \
                                                     # the file path
    table_name = db_file_name.split(".")[0]
    df.to_sql(table_name, engine, index=False, if_exists = 'replace')
    
#     engine = create_engine('sqlite:///' + database_filename)
#     df.to_sql('messages_disaster', engine, index=False)
   


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath =  sys.argv[1:]

        print(messages_filepath)
        print(categories_filepath)
        print(database_filepath)
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