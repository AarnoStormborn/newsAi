import os
import re
import sys
import yaml
import json
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from box import ConfigBox

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import (classification_report, 
                             accuracy_score, f1_score,
                             confusion_matrix)

warnings.filterwarnings('ignore')

def read_config(filepath):
    with open(filepath) as f:
        data = yaml.safe_load(f)
    data = ConfigBox(data)
    return data    

def save_results(filepath, model_type, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    if model_type == 'classification':
        f1 = f1_score(y_true, y_pred, average='micro')
    else:
        f1 = f1_score(y_true, y_pred)

    pd.DataFrame({
        'accuracy_score': [accuracy],
        'f1_score': [f1]
    }).to_csv(os.path.join(filepath, f"{model_type}_results.csv"))
    logging.info(f"Saved Results: {model_type}")

    clf_report = classification_report(y_true, y_pred, output_dict=True)
    with open(os.path.join(filepath, f"{model_type}_classification_report.json"), "w") as f:
        json.dump(clf_report, f, indent=4)

    logging.info(f"Saved Classification Report: {model_type}")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, cmap='Blues', cbar=False, annot=True, fmt='.3g')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(filepath, f"{model_type}_confusion_matrix.png"))
    logging.info(f"Saved Confusion Matrix: {model_type}")

def sentiment(dataframe):

    def sentiment_score(text):
        return TextBlob(text).sentiment.polarity 

    try:
        if 'sentiment_score' not in dataframe.columns:
            dataframe['sentiment_score'] = dataframe.news_article.apply(sentiment_score)
            threshold = dataframe.sentiment_score.quantile(q=0.5)
            if 'sentiment' not in dataframe.columns:
                dataframe['sentiment'] = 0
                dataframe['sentiment'][dataframe['sentiment_score']>=threshold] = 1
                dataframe.drop(['sentiment_score'], axis=1, inplace=True)
        
        dataframe = dataframe.sample(frac=1)

        return dataframe

    except Exception as e:
        logging.info(CustomException(e, sys))

def cleanse(text):

    try:
        text = " ".join(x.lower() for x in text.split())
        text = re.sub('[^a-zA-Z0-9-]+', ' ', text)
        stop_words = set(stopwords.words('english'))
        text = " ".join(word for word in text.split() if word not in stop_words)
        lemmatizer = WordNetLemmatizer()
        text = " ".join(lemmatizer.lemmatize(word) for word in text.split())
        
        return text
    
    except Exception as e:
        logging.info(CustomException(e, sys))


