import numpy as np
import pandas as pd 
from exception.exception import customexception
from logger.logging import logging 
import os
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet

from dataclasses import dataclass 

@dataclass
class ModelTrainingConfig:
    trained_model_path= os.path.join("artifacts", "model.pkl")
    

class ModelTraining:
    def __init__(self):
        self.model_path = ModelTrainingConfig()


    def initiate_model_training(self, train_array, test_array):
        logging.info("Setting Model Training")

        try:
            logging.info("Train_test_split")
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_path.trained_model_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)


            return (
                self.data_path.train_data,
                self.data_path.test_data
            )

           

        except Exception as e:
            logging.info("Error loading dataset")
            raise customexception(e, sys)
