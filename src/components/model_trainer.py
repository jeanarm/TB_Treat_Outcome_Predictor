import os
import sys
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("data","model.pkl")
    label_encoder_file_path =os.path.join("data","labelencoder.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config =ModelTrainerConfig()

    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):
        try:    
            models = {
                'KNeighbors': KNeighborsClassifier(),
                'DecisionTree': DecisionTreeClassifier(),
                'RandomForest': RandomForestClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'SVM': SVC(),
                'LogisticRegression': LogisticRegression(),
                'Ridge': RidgeClassifier()
            }
            label_encoder =LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.fit_transform(y_test)
          
            param_grid = {
                'KNeighbors': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                
                'DecisionTree': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                
                'RandomForest': {
                    'n_estimators': [100, 200],
                    'max_features': ['sqrt', 'log2']
                },
                
                'AdaBoostClassifier': {
                'base_estimator': [DecisionTreeClassifier(max_depth=1), RandomForestClassifier(max_depth=1)],
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 1.0]
                },

                            
                'SVM': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.1, 1.0]
                },
                
                'LogisticRegression': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2']
                },
                
                'Ridge': {
                    'alpha': [0.1, 1.0, 10.0]
                }
            }


            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train_encoded, X_test=X_test,y_test=y_test_encoded, models = models,params=param_grid)

            #Get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #Get Best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
               raise CustomException("No best model Found")
            logging.info(f"Best Found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            save_object(
                file_path=self.model_trainer_config.label_encoder_file_path,
                obj= label_encoder
            )
            
            predicted_encoded = best_model.predict(X_test)
            predicted_decoded = label_encoder.inverse_transform(predicted_encoded)
            #print("Predicted True value:", predicted_decoded)
            accuracy = accuracy_score(y_test_encoded,predicted_encoded)
            precision = precision_score(y_test_encoded,predicted_encoded, average='weighted')
            recall= recall_score(y_test_encoded,predicted_encoded, average='weighted')
            f1 = f1_score(y_test_encoded,predicted_encoded, average='weighted')
            print("===============================================")
            print("Best model After params tuning:",best_model_name)
            print("Confusion matrix")
            print(confusion_matrix(y_test_encoded, predicted_encoded))
                                                                       
            return {"Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1":f1}
        except Exception as e:
            raise CustomException(e,sys)