import os
import sys
from exception.exception import customexception
from utils.utils import save_object, load_object
from logger.logging import logging
import pandas as pd




class ModelPredict:
    def __init__(self):
        pass

    def predictPipeline(self,features):
        try:
            preprocessor_path  = os.path.join("artifacts", "processor.pkl")
            model_path  = os.path.join("artifacts", "model.pkl")

            preproccessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data = preproccessor.transform(features)
            pred_class = model.predict(data)
            pred_class = round(pred_class[0],2)

            return pred_class

            
        except Exception as e:
            raise customexception(e,sys)

class CustomData:
    def __init__(self, carat:float, depth:float, table:float,x:float,y:float,z:float,cut:str,color:str,clarity:str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
        
    def get_data_to_dataframe(self):
        try:
            data = {'carat' : [self.carat],
                    'depth' : [self.depth],
                    'table' : [self.table],
                    'x'     : [self.x],
                    'y'     : [self.y],
                    'z'     : [self.z],
                    'cut'   : [self.cut],
                    'color' : [self.color],
                    'clarity': [self.clarity]}

            df= pd.DataFrame(data)
            logging.info("Dataset exported")
            return df

        except Exception as e:
            logging.info("error taking prediction dataset")
            raise customexception(e,sys)


    