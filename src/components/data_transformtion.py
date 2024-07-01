import numpy as np
import pandas as pd 
from exception.exception import customexception
from logger.logging import logging 
import os
import sys
from sklearn.model_selection import train_test_split

from dataclasses import dataclass 
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from utils.utils import save_object

@dataclass
class DataTransformationConfig:
    processor_obj_path = os.path.join("artifacts", "processor.pkl")

    
class DataTransformation:
    def __init__(self):
        self.processor_path = DataTransformationConfig()

    def get_data_preprocessor(self):
        logging.info("Data Transformation Started")

        try:
            
            cat_column = ["cut","color","clarity"]
            num_column = ["carat", "depth", "table", "x", "y", "z"]

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Initiating pipeline......")


            numerical_pipeline = Pipeline(

                steps = [("Imputer", SimpleImputer(strategy = 'median')),
                    ("Standard scaler", StandardScaler())]
            )


            categorical_pipeline = Pipeline(

                steps = [("Imputer", SimpleImputer(strategy="most_frequent")),
                ("Encoding", OrdinalEncoder(categories=[cut_categories, color_categories,clarity_categories])),

                    ("Standard scaler", StandardScaler())]
            )

            preprocessor= ColumnTransformer(
                [("num_column", numerical_pipeline, num_column),
                ("cat_column", categorical_pipeline, cat_column)

                ]
            )

            return preprocessor

            
        except Exception as e:
            logging.info("Error preprocessing dataset dataset")
            raise customexception(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Initialiing data transformation...")

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        logging.info("Inializing processor...")

        preprocessor_obj = self.get_data_preprocessor()

        target = 'price'

        drop_features = ["id", target]

        logging.info("Dropping unwanted features...")

        train_features = train_data.drop(columns=drop_features, axis=1)
        test_features = test_data.drop(columns=drop_features, axis = 1)

        train_target = train_data[target]
        test_target = test_data[target]

        logging.info("data transformation begin...")

        training_data =preprocessor_obj.fit_transform(train_features )
        testing_data =preprocessor_obj.transform( test_features )

        logging.info("data transformation end...")

        logging.info("data transformation concatenation...")

        train_arr = np.c_[training_data, np.array(train_target) ]
        test_arr  = np.c_[testing_data, np.array(test_target)]


       

       

        logging.info("Data transformed...")
        save_object(file_path= self.processor_path.processor_obj_path, obj= preprocessor_obj)


        return ( train_arr, test_arr)
