import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os,sys
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifact","train.csv")
    test_data_path=os.path.join("artifact","test.csv")
    raw_data_path=os.path.join("artifact","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("/Applications/Study Material/Data Science/mlproject/Data/stud.csv")
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            train_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Inmgestion of the data iss completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if (__name__=="__main__"):
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,processor_obj=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    score=model_trainer.initiate_model_training(train_arr,test_arr)


