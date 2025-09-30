Sentiment Analysis on Twitter Data
This project demonstrates a complete machine learning workflow for classifying the sentiment of tweets. Using a large dataset of 1.6 million tweets, the model is trained to accurately predict whether a given text expresses a positive or negative sentiment. This project showcases proficiency in data preprocessing, machine learning model building, and evaluation.

Key Features
Large-Scale Data Handling: Processes a large dataset of over a million tweets.

Text Preprocessing: Includes robust steps to clean raw text data, such as removing URLs, punctuation, and stopwords.

Machine Learning Model: Builds and trains a Naive Bayes Classifier for sentiment prediction.

Model Evaluation: Evaluates the model's performance using standard metrics.

Live Prediction: A function to predict the sentiment of new, unseen text.

Technologies Used
Python: The core programming language.

Pandas: For efficient data manipulation and analysis.

Scikit-learn: The primary library for machine learning, including model building and evaluation.

NLTK (Natural Language Toolkit): For advanced text processing tasks like tokenization and stopword removal.

Kaggle API: To programmatically download the dataset.

Google Colab: The development environment for this project.

Getting Started
This project was developed in a Google Colab environment. To run the notebook and reproduce the results, follow the steps below.

1. Setup and Library Installation
First, install the necessary libraries. This is a crucial step to set up the environment.

Python

!pip install pandas scikit-learn nltk
!pip install -q kaggle
2. Data Acquisition (Kaggle API)
The project uses the Sentiment140 dataset. You need to download it using your Kaggle API key.

Go to your Kaggle account and create an API token to get kaggle.json.

Upload the kaggle.json file to your Colab session.

Run the following code to set up your environment and download the dataset.

Python

from google.colab import files
files.upload()

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d kazanova/sentiment140

import zipfile
import os
with zipfile.ZipFile('sentiment140.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
3. Data Loading and Cleaning
Load the dataset into a Pandas DataFrame and apply a cleaning function to remove unwanted characters and stopwords from the tweets.

Python

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')
# The following is needed for some older versions of NLTK
try:
    nltk.download('punkt_tab')
except Exception:
    pass

df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

def clean_text(text):
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

df['text'] = df['text'].apply(clean_text)
4. Model Training and Evaluation
Convert the text data into a numerical format using TfidfVectorizer, then train a Multinomial Naive Bayes model and evaluate its performance.

Python

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

X = df['text']
y = df['sentiment'].map({0: 'negative', 4: 'positive'})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)
5. Making Predictions on New Data
Use the trained model to predict the sentiment of custom sentences.

Python

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    return prediction

test_sentences = [
    "I am very happy with this project!",
    "This is a terrible idea.",
    "The weather is just fine."
]

for sentence in test_sentences:
    sentiment = predict_sentiment(sentence)
    print(f"Sentence: '{sentence}' -> Predicted Sentiment: '{sentiment}'")
