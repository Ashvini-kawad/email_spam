import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from log_file import log_function
logger=log_function('data_ingestion')

def data_load(data_url:str) -> pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug('data loaded from %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error(' failed to parse the data:%s',e)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading data:%s',e)
        raise


def data_clean(df:pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error("column name is missing in the data:%s",e)
        raise
    except Exception as e:
        logger.error("unexpected error occured while dropping columns: %s",e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug("train and text data saved: %s",raw_data_path)
    except Exception as e:
        logger.error("Unexpected error ocuured while saving data:%s",e)
        raise

def main():
    try:
        data_url='Data\spam.csv'
        test_size=0.2
        df=data_load(data_url)
        process_df=data_clean(df)
        train_data, test_data = train_test_split(process_df, test_size=test_size, random_state=2)
        save_data(train_data,test_data,data_path='./Data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process:%s',e)
        print(f'Error:{e}')

if __name__ == '__main__':
    main()


        