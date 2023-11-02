import os
import sys
import time
import joblib
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import read_config
from src.constant import PARAMS_FILE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    root_dir: str
    preprocessor_path: str

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config = config

    def model_trainer(self, train_set:pd.DataFrame) -> None:
        try:
            root_dir = self.config.root_dir
            os.makedirs(root_dir, exist_ok=True)

            X_train, y_train_category, y_train_sentiment = train_set

            logging.info("Starting Classification Model Training...")
            start = time.time()
            rfc = RandomForestClassifier(random_state=42)
            rfc.fit(X_train, y_train_category)            

            end = time.time() - start
            logging.info(f"Classifiaction Model Training Complete. Time taken: {end:.2f} seconds")

            joblib.dump(rfc, os.path.join(root_dir, "classifier.pkl"))
            logging.info("Best Classification Model Saved")

            logging.info("Starting Sentiment Model Training...")
            rfc_sent = RandomForestClassifier(random_state=42)
            rfc_sent.fit(X_train, y_train_sentiment)

            end = time.time() - start

            logging.info(f"Sentiment Model Training Complete. Time taken: {end:.2f} seconds")
            joblib.dump(rfc_sent, os.path.join(root_dir, "sentiment.pkl"))
            logging.info("Best Sentiment Model Saved")

        except Exception as e:
            logging.error(CustomException(e,sys))

