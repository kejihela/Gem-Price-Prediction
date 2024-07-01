import numpy as np
import pandas as pd 
from exception.exception import customexception
from logger.logging import logging 
import os
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path

from dataclasses import dataclass 

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts", "raw_data.csv")
    train_data_path = os.path.join("artifacts", "train_data.csv")
    test_data_path = os.path.join("artifacts", "test_data.csv")

class DataIngestion:
    def __init__(self):
        self.data_path = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")

        try:
            logging.info("Loading dataset")
            data = pd.read_csv("https://raw.githubusercontent.com/kejihela/End-to-End-MLOps/master/src/componets/train.csv")
            os.makedirs(os.path.dirname(os.path.join(self.data_path.raw_data_path)),exist_ok=True)
          

            data.to_csv(self.data_path.raw_data_path,index = False)

            logging.info("saving the raw dataset")

            logging.info("Splitting data into train and test set")

            train_data, test_data = train_test_split(data, test_size=0.3)


            train_data.to_csv(self.data_path.train_data_path, index = False)
            logging.info("saving the training dataset")


            test_data.to_csv(self.data_path.test_data_path, index = False)
            logging.info("saving the test dataset")

            return (
                self.data_path.train_data_path,
                self.data_path.test_data_path
            )

           

        except Exception as e:
            logging.info("Error loading dataset")
            raise customexception(e, sys)

