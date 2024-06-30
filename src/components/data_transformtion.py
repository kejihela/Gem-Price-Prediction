import numpy as np
import pandas as pd 
from src.exception.exception import customexception
from src.logger.logging import logging 
import os
import sys
from sklearn.model_selection import train_test_split

from dataclasses import dataclass 
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.make_pipeline import pipeline
from utils.utils import save_object

@dataclass
class DataTransformationConfig:
    processor_obj_path = os.path.join("artifacts", "processor.pkl")

    
class DataTransformation:
    def __init__(self):
        self.processor_path = DataTransformationConfig()

    def get_data_preprocessor():
        logging.INFO("Data Transformation Started")

        try:
            
            cat_column = ["cut","color","clarity"]
            num_column = ["carat", "depth", "table", "x", "y", "z"]

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.INFO("Initiating pipeline......")


            numerical_pipeline = pipeline(

                step = [("Imputer", SimpleImputer(strategy = 'median')),
                    ("Standard scaler", StandardScaler())]
            )


            categorical_pipeline = pipeline(

                step = [("Imputer", SimpleImputer(strategy="most_frequent")),
                ("Encoding", OrdinalEncoder(categories=(" cut_categories", "color_categories","clarity_categories")))

                    ("Standard scaler", StandardScaler())]
            )

            preprocessing = ColumnTransformer(
                [("num_column", numerical_pipeline, num_column),
                ("cat_column", categorical_pipeline, cat_column)

                ]
            )

            return preprocessor

            
    except Exception as e:
        logging.INFO("Error preprocessing dataset dataset")
        raise customexception(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        logging.INFO("Initialiing data transformation...")

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        preprocessor_obj = get_data_preprocessor()

        target = 'price'

        drop_features = ["id", target]

        train_features = train_data.drop(drop_features, axis=1)
        test_features = test_data.drop(drop_features, axis = 1)

        train_target = train_data[target]
        test_target = test_data[target]


        training_data = np.c_[train_features, np.array(train_target)]
        testing_data = np.c_[test_features, np.array(test_target)]

        train_arr =preprocessor_obj.fit_transform(training_data)
        test_arr =preprocessor_obj.transform(testing_data)


        save_object(file_name= self.processor_path.processor_obj_path, obj= preprocessor_obj)


        return ( train_arr, test_arr)
