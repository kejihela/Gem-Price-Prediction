import os
import sys
from logger.logging import logging
from utils.utils import save_object, load_object
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import mlflow
import mlflow.sklearn
from exception.exception import customexception
import numpy as np






class ModelEvaluation:
    def __init__(self):
        logging.info("starting model evaluation")


    def model_eval(self, y_test, pred):

        try:
            rmse = np.sqrt(mean_squared_error(y_test,pred))
            mse = mean_squared_error(y_test,pred)
            r2 = r2_score(y_test,pred)

            return rmse, mse, r2

        except Exception as e:
            logging.info("error evaluating model")
            raise customexception(e,sys)


    def initiate_model_evaluation(self, train_arr, test_arr):
        logging.info("loading model for evaluation")
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            model =  load_object(model_path)

            X_test, y_test = (test_arr[:,:-1], test_arr[:,-1])

            pred = model.predict(X_test)

            rmse, mse, r2 = self.model_eval(y_test, pred)

            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("Diamond price prediction")
            with mlflow.start_run():

                mlflow.log_metric('rmse', rmse)
                mlflow.log_metric('mse', mse)
                mlflow.log_metric('r2', r2)

                mlflow.log_param("train_data_path", "artifacts/train_data.csv")
                mlflow.log_param("test_data_path", "artifacts/test_data.csv")

                mlflow.sklearn.log_model( model, "model")

            logging.info("Done with pipeline")

        except Exception as e:
            logging.info("error evaluating model")
            raise customexception(e,sys)
