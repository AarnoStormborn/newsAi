import os
import sys
import zipfile
from urllib import request
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_URL: str
    local_file: str

class DataIngestion:

    def __init__(self, config:DataIngestionConfig):
        self.config = config

    def download_zip_file(self) -> None:

        try:
            os.makedirs(self.config.root_dir, exist_ok=True)
            if not os.path.exists(self.config.local_file):
                filename, _ = request.urlretrieve(
                    url = self.config.source_URL,
                    filename = self.config.local_file
                )
                logging.info(f"{filename} downloaded")
            else:
                logging.info("File already exists")
        except Exception as e:
            logging.error(CustomException(e,sys))    

    def extract_zip_file(self) -> None:
        try:
            with zipfile.ZipFile(self.config.local_file, 'r') as zip_ref:
                zip_ref.extractall(self.config.root_dir)
        except Exception as e:
            logging.error(CustomException(e, sys))

