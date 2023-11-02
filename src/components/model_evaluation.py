import os
import sys
import json
import joblib
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_results

@dataclass
class ModelEvaluationConfig:
    root_dir: str
    classification_model_path: str
    sentiment_model_path: str

class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config

    def model_evaluator(self, test_set:pd.DataFrame) -> None:
        try:
            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)

            X_test, y_test_category, y_test_sentiment = test_set

            logging.info("Loading Classification Model")
            classifier = joblib.load(self.config.classification_model_path)
            y_pred_category = classifier.predict(X_test)

            sentiment = joblib.load(self.config.sentiment_model_path)
            y_pred_sentiment = sentiment.predict(X_test)

            save_results(root_dir, 'classification', y_test_category, y_pred_category)
            save_results(root_dir, 'sentiment', y_test_sentiment, y_pred_sentiment)

            logging.info("Results saved")

        except Exception as e:
            logging.error(CustomException(e, sys))
