import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:

            model_path = 'data/model.pkl'
            preprocessor_path = 'data/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds= model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:

    def __init__ (self,
                  controlEnd:str,
                  controlAfter2:str,
                  confirmationMethod:str,
                  controlAfter5:str,
                  siteofdesease:str,
                  treatmenthistory:str,
                  smeaResult:str,
                  genexpertResult:str,
                  hospital:str,
                  weightend:float):
        self.controlEnd = controlEnd
        self.controlAfter2 = controlAfter2
        self.confirmationMethod = confirmationMethod
        self.controlAfter5 = controlAfter5
        self.siteofdesease = siteofdesease
        self.genexpertResult = genexpertResult
        self.smeaResult = smeaResult
        self.treatmenthistory = treatmenthistory
        self.weightend = weightend
        self.hospital = hospital


    def get_data_as_data_frame(self):

        try:
            custom_data_input_dict = {
                "Control After TB Treatment":[self.controlEnd],
                "Control After 2 month":[self.controlAfter2],
                "Control After 5 month":[self.controlAfter5],
                "Site of disease":[self.siteofdesease],
                "GeneXpert results - MTB":[self.genexpertResult],
                "Method of TB confirmation":[self.confirmationMethod],
                "Previous treatment history":[self.treatmenthistory],
                "Smear specimen result":[self.smeaResult],
                "Weight after TB Treatment(kg)":[self.weightend],
                "Hospital":[self.hospital]

            }
            return pd.DataFrame(custom_data_input_dict)


        except Exception as e:
            raise CustomException(e,sys)
