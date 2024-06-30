import numpy as np
import pandas as pd 
from src.exception.exception import customexception
from src.logger.logging import logging 
import os
import sys
from sklearn.model_selection import train_test_split

from dataclasses import dataclass 

@dataclass
class DataIngestionConfig:
    raw_data = os.path.join("artifacts", "raw_data")
    train_data = os.path.join("artifacts", "train_data")
    test_data = os.path.join("artifacts", "test_data")

class DataIngestion:
    def __init__(self):
        self.data_path = DataIngestionConfig()


    def initiate_data_ingestion():
        logging.INFO("Data Ingestion Started")

        try:
            logging.INFO("Loading dataset")
            data = pd.read_csv("https://raw.githubusercontent.com/kejihela/End-to-End-MLOps/master/src/componets/train.csv")
            os.path.makedirs(os.path.dirname(os.path.join(self.data_path.raw_data)),exist_ok=True)

            data.to_csv(self.data_path.raw_data,index = False)

            logging.INFO("saving the raw dataset")

            logging.INFO("Splitting data into train and test set")

            train_data, test_data = train_test_split(data, test_size=0.3)

            os.path.makedirs(os.path.dirname(self.data_path.train_data),exist_ok=True)

            train_data.to_csv(self.data_path.train_data, index = False)
            logging.INFO("saving the training dataset")

            os.path.makedirs(os.path.dirname(self.data_path.test_data),exist_ok=True)

            test_data.to_csv(self.data_path.test_data, index = False)
            logging.INFO("saving the test dataset")

            return (
                self.data_path.train_data,
                self.data_path.test_data
            )

           

    except Exception as e:
        logging.INFO("Error loading dataset")
        raise customexception(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()

    obj.initiate_data_ingestion()