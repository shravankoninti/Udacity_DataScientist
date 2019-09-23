# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support,accuracy_score, precision_score, recall_score, f1_score


def load_data(database_filepath):
    """
    The data load file path which has cleaned dataframe

    Args: 
    
    database_filepath (str): The input file with the cleaned datafile - SQL file
    

    Returns:
    X: (numpy.ndarray). Disaster messages.
    Y: (numpy.ndarray). Disaster categories for each messages.
    category_name: (list). Disaster category names.  

    """
    # load data from database 
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
 
    db_file_name = database_filepath.split("/")[-1] # extract file name from \
                                                     # the file path
  
    table_name = db_file_name.split(".")[0]
 
    df = pd.read_sql_table(table_name, con=engine)
    
    categories = df.columns[4:]
    
    X = df["message"]
    y = df.drop(['id', 'message', 'original', 'genre','categories'], axis = 1)

    return X, y, categories

def tokenize(text):
    """
    Tokenize text (a disaster message).
    
    Args:
        text (String) : A disaster message/text
        lemmatizer: nltk.stem.Lemmatizer.
    Returns:
        list -  It contains tokens as each word
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Function: build model that consist of pipeline parameters
    
    Args:
    
      N/A
      
    Return:
    
      cv(model): Grid Search CV model 
      
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
    'tfidf__use_idf': (True, False), 
    'clf__estimator__n_estimators': [50, 75, 100], 
    'clf__estimator__min_samples_split': [2, 6, 8]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return cv 

def display_results(y_true, y_pred, categories):
    """
    Function: Display the results of accuracy , f1-score and other metrics from the model
    
    Args:
    
      y_true (np.array) : Actual values of model output
      
      y_pred (np.array) : Predicted values of model output
      
      categories (list) : List of column values in category
      
    Return:
    
      N/A 
      
    """
    for i in range(0, len(categories)):
        print(categories[i])
        print("\tAccuracy: {:.4f}\t\t% Precision: {:.4f}\t\t% Recall: {:.4f}\t\t% F1_score: {:.4f}".format(
            accuracy_score(y_true.loc[:, categories[i]], y_pred.loc[:, categories[i]]),
            precision_score(y_true.loc[:, categories[i]], y_pred.loc[:, categories[i]], average='weighted'),
            recall_score(y_true.loc[:, categories[i]], y_pred.loc[:, categories[i]], average='weighted'),
            f1_score(y_true.loc[:, categories[i]], y_pred.loc[:, categories[i]], average='weighted')    
            
        ))
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function: Evaluates the model on the test data
    
    Args:
    
      Model (model) : The machine learning algorithm implemented as model
      
      X_test (np.array) : The input parameters which has independent variables information
      
      Y_test (np.array) : The input parameters with the dependent variable
      
      category_names (list) : The list of categroy or column names
      
    Return
      cv(model): Grid Search model 
      
    """
    
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns = Y_test.columns)
    display_results(Y_test, y_pred, category_names)

def save_model(model, model_filepath):
    
#     """
#      Save the final model
     
#     Args:
#         model: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
#         model_filepath: String. Trained model is saved as pickle into this file.
        
#     """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
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