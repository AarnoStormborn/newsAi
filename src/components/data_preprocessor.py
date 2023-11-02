import os
import sys
import joblib
import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import read_config, cleanse 
from src.constant import SCHEMA_FILE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

@dataclass
class DataPreprocessorConfig:
    root_dir: str
    train_data_path: str
    test_data_path: str

class DataPreprocessor:
    def __init__(self, config:DataPreprocessorConfig):
        self.config = config        

    def get_preprocessor(self) -> Pipeline:
        try:
            preprocessor = Pipeline([
                ("vectorizer", TfidfVectorizer())
            ])

            logging.info("Preprocessor Pipeline Created")

            return preprocessor
        
        except Exception as e:
            logging.error(CustomException(e, sys))
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)

            schema = read_config(SCHEMA_FILE)
            feature = schema.features
            category = schema.news_category
            sentiment = schema.sentiment

            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            X_train, y_train_category, y_train_sentiment = train_df[feature], train_df[category], train_df[sentiment]
            X_test, y_test_category, y_test_sentiment = test_df[feature], test_df[category], test_df[sentiment]
            
            X_train_cleaned = X_train.apply(cleanse)
            X_test_cleaned = X_test.apply(cleanse)

            logging.info("Loading Preprocessor")

            preprocessor = self.get_preprocessor()
            preprocessor.fit(X_train_cleaned)

            label_encoder = LabelEncoder()
            label_encoder.fit(y_train_category)

            X_train_transformed = preprocessor.transform(X_train_cleaned)
            X_test_transformed = preprocessor.transform(X_test_cleaned)

            y_train_category_encoded = label_encoder.transform(y_train_category)
            y_test_category_encoded = label_encoder.transform(y_test_category)

            logging.info("Data Succesfully Preprocessed")

            train_data_transformed = X_train_transformed, y_train_category_encoded, y_train_sentiment
            test_data_tranformed = X_test_transformed, y_test_category_encoded, y_test_sentiment

            joblib.dump(preprocessor, os.path.join(root_dir, "vectorizer.pkl"))
            joblib.dump(label_encoder, os.path.join(root_dir, "encoder.pkl"))

            logging.info("Preprocessor Object Saved")

            return (train_data_transformed, test_data_tranformed)
        
        except Exception as e:
            logging.error(CustomException(e, sys))





