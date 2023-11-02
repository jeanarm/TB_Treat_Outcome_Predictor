import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('data',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["Weight after TB Treatment(kg)"]
            categorical_columns = [ "Site of disease", "GeneXpert results - MTB", "Method of TB confirmation",
                        "Control After TB Treatment", "Control After 2 month", "Control After 5 month",
                        "Smear specimen result", "Previous treatment history"]



            num_pipeline =Pipeline(
                steps=[
                    # ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    # ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    #("scaler", StandardScaler(with_mean=False))
                ]
            )


            logging.info(f"Numerical Columns: {numerical_columns}")
            logging.info(f"Categorical Columns: {categorical_columns}")
            preprocessor = ColumnTransformer(
                    [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline,categorical_columns)
                    ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test completed")

            logging.info("obtening preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "Treatment outcome"
            numerical_columns = ["Weight after TB Treatment(kg)"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
             
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."

            )
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
        
            X_train = input_feature_train_arr
            X_test = input_feature_test_arr
            y_train = target_feature_train_df
            y_test= target_feature_test_df

            logging.info(f"Saved preprocessing objects.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                X_train,
                X_test,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path

            )



        except Exception as e:

         raise CustomException(e,sys)
