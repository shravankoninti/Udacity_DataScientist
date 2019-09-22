## Data Scientist Nanodegree

## Project: Disaster Response Pipeline

## Table of Contents

1. [Installation](##Installation)
2. [Project Overview](##Project-Overview)
3. [File Descriptions](##File-Descriptions)
4. [Results](##Results)
5. [Licensing, Authors, and Acknowledgements](##Licensing\, Authors\,-and-Acknowledgements)
6. [Instructions](##Instructions)


## Installation
Beyond the Anaconda distribution of Python, the following packages need to be installed:

1. numpy
2. pandas
3. pickle 
4. punkt
5. wordnet
6. stopwords


## Project Overview

In this project, I have applied data engineering & machine learning skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. This project majorly uses text analytics.

data directory contains a data set which are real messages that were sent during disaster events. I have created a machine learning pipeline to categorize these events so that appropriate disaster relief agency can be reached out for help.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

Here are a few screenshots of the web app.

<img width="940" alt="pre_1" src="https://user-images.githubusercontent.com/6191291/65388669-af12e100-dd6b-11e9-942b-0356055479c0.PNG">

<img width="945" alt="pre_2" src="https://user-images.githubusercontent.com/6191291/65388675-c782fb80-dd6b-11e9-965c-0860b5d5175f.PNG">

<img width="947" alt="pre_3" src="https://user-images.githubusercontent.com/6191291/65388686-eda89b80-dd6b-11e9-92e0-ffffd8c0b5bb.PNG">

## File Descriptions

There are three main folders:

### *data*
disaster_categories.csv: dataset including all the categories
disaster_messages.csv: dataset including all the messages
process_data.py: ETL pipeline scripts to read, clean, and save data into a database
DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
### *models*
train_classifier.py: machine learning pipeline scripts to train and export a classifier
classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer
### *app*
run.py: Flask file to run the web application
templates contains html file for the web application

## Results

There are 3 workflows that are followed here:

1. An ETL pipleline was built to read data from two csv files (disaster_categories.csv, disaster_messages.csv), clean data, and save data into a SQLite database.
2. A machine learning pipepline was developed to train a classifier to perform multi-output classification on the 36 categories in the dataset.
3. A Flask app was created to show data visualization and classify the message that user enters on the web page.

some of the screenshots of the results are shown below:


<img width="948" alt="result-1" src="https://user-images.githubusercontent.com/6191291/65392395-2b201f80-dd92-11e9-9cd8-d514c78f7fad.PNG">

<img width="945" alt="result-2" src="https://user-images.githubusercontent.com/6191291/65392402-37a47800-dd92-11e9-881c-7fe21349f69a.PNG">

<img width="906" alt="result-3" src="https://user-images.githubusercontent.com/6191291/65392408-4854ee00-dd92-11e9-81d5-8fee23e28276.PNG">

<img width="945" alt="result-4" src="https://user-images.githubusercontent.com/6191291/65392415-54d94680-dd92-11e9-8531-a755d29de9b2.PNG">

## Licensing, Authors, Acknowledgements
Udacity has provided starter codes and FigureEight has provided the disaster messages data for analysis. 
Please refer to Udacity Terms of Service for further information.

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv  data/disaster_categories.csv data/DisasterResponse.db
   - To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/
