import os
import sys
import joblib

from pandas import Series
from dataclasses import dataclass
from src.utils import cleanse
from src.logger import logging
from src.exception import CustomException

@dataclass
class PredictionConfig:
    preprocessor_path: str
    encoder_path: str
    classification_model_path: str
    sentiment_model_path: str

class PredictionPipeline:

    def __init__(self, config:PredictionConfig):
        self.config = config
        self.vectorizer = joblib.load(self.config.preprocessor_path)
        self.encoder = joblib.load(self.config.encoder_path)
        self.classifier = joblib.load(self.config.classification_model_path)
        self.sentiment = joblib.load(self.config.sentiment_model_path)

    def predict(self, data:Series) -> str:

        try:
            logging.info("Prediction Data Obtained")
            data_cleaned = data.apply(cleanse)

            prediction_data = self.vectorizer.transform(data_cleaned)
            logging.info("Prediction Data Preprocessed")

            category_prediction = self.classifier.predict(prediction_data)
            category = self.encoder.inverse_transform(category_prediction)

            sentiment = self.sentiment.predict(prediction_data)

            logging.info("Model Prediction Complete")

            predictions = {
                'category': category,
                'sentiment_probability': sentiment
            }

            return predictions

        except Exception as e:
            logging.error(CustomException(e,sys))  
