from src.components.data_ingestion import DataIngestion
from src.components.data_transformtion import DataTransformation
from src.components.model_trainer import ModelTraining
from src.components.model_evaluation import model_evaluation

from src.logger.logging import logging
from src.exception.exception import customexception
import pandas as pd
import numpy as np
import os
import sys

data_obj = DataIngestion()
training_path, testing_path = data_obj.initiate_data_ingestion()

transform_obj = DataTransformation()
train, test = transform_obj.initiate_data_transformation(training_path, testing_path )

model_obj = ModelTraining()
model_obj.initiate_model_training(train, test)


