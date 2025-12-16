import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from log_file import log_function
from dvclive import Live
import yaml
logger=log_function("model_evalution")

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path,'rb') as file:
            model=pickle.load(file)
        logger.debug("model is load  from %s",file_path)
        return model
    except FileExistsError as e:
        logger.error(" file is not fount %s",e)
        raise
    except Exception as e:
        logger.error("unexpected error is occured while loading model %s",e)
        raise

def load_data(file_path):
    "loading the data from the file_path"
    try:
        df=pd.read_csv(file_path)
        logger.debug(" data is loaded from %s",file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def model_evalution(clf,X_test,y_test):
    try:
        y_pred=clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
           
        }

        logger.debug("Model evalution metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error("Unexpected error is occured while model evalution %s",e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        params=load_params('params.yaml')
        clf=load_model("models\model.pkl")
        test_data=load_data(r"Data\processed\test_tfidf.csv")
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params)

        metrics_dict=model_evalution(clf,X_test,y_test)
        save_metrics(metrics_dict,"reports\metrics.json")
    
    except Exception as e:
        logger.error("failed to complete model evalution process %s",e)
        raise

if __name__=='__main__':
    main()

    
