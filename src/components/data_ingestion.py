import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_apth = os.path.join('artifacts','train.csv')
    test_data_apth = os.path.join('artifacts','test.csv')
    raw_data_apth = os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enterted the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset from data source as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_apth),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_apth,index=False,header=True)

            logging.info("Train Test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_apth,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_apth,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_apth,
                self.ingestion_config.test_data_apth
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    print("Data Ingestion Started..")
    obj = DataIngestion()
    train_data ,test_data =obj.initiate_data_ingestion()
    print("Data Ingestion Done")

    print("Data Transformation Started...")
    data_transformation = DataTransformation()
    train_array,test_array,_=data_transformation.initiate_data_transformation(train_data,test_data)
    print("Data Transformation Succefully completed")

    print("Model Training Started...")
    modelTrainer = ModelTrainer()
    r2_square=modelTrainer.initiate_model_trainer(train_array,test_array)
    print("Model Training Succefully completed.")
    print(r2_square)

