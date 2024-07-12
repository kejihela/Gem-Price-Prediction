from components.data_ingestion import DataIngestion
from components.data_transformtion import DataTransformation
from components.model_trainer import ModelTraining
from components.model_evaluation import  ModelEvaluation

from logger.logging import logging
from exception.exception import customexception
import pandas as pd
import numpy as np
import os
import sys
import mlflow

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_obj = DataIngestion()
            training_path, testing_path = data_obj.initiate_data_ingestion()
            return training_path, testing_path
        except Exception as e:
            raise customexception(e,sys)

    def start_data_transformtion(self):
        try:
            transform_obj = DataTransformation()
            train, test = transform_obj.initiate_data_transformation(training_path, testing_path )
            return train, test
        except Exception as e:
            raise customexception(e,sys)

    def start_model_training(self):
        try:
            model_obj = ModelTraining()
            model_obj.initiate_model_training(train, test)
        except Exception as e:
            raise customexception(e,sys)


    def start_training(self):
        try:
           trainpath, testpath =  self.start_data_ingestion()
           train,test = self.start_data_transformtion()
           self.start_model_training(train,test)

        except Exception as e:
            raise customexception(e, sys)


#model_evaluate = ModelEvaluation()

#model_evaluate.initiate_model_evaluation(train, test )



