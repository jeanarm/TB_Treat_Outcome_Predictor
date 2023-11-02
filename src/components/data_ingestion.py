import os
import sys
sys.path.append('/Users/armel/Documents/TB_Flask_Predictor_App')
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.components.data_transformation import DataTransformation

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data',"train.csv")
    test_data_path: str = os.path.join('data', "test.csv")
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config =  DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or components")
        try:
            df = pd.read_csv('data/dataset.csv')
            logging.info('read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header= True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header= True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj =   DataIngestion()
    obj.initiate_data_ingestion()
    train_data,test_data =  obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    X_train,X_test,y_train,y_test,_= data_transformation.initiate_data_transformation(train_data,test_data)
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(X_train,X_test,y_train,y_test))
    print("===============================================")