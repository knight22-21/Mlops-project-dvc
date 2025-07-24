import os
import json
import pickle
import numpy as np  
import pandas as pd
import logging
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Logging setup
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_pickle(path):
    """Load a pickle file."""
    try:
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        logger.debug(f"Loaded pickle file from {path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading pickle file from {path}: {e}")
        raise


def load_data(path):
    try:
        df = pd.read_csv(path)
        logger.debug(f"Loaded test data from {path} with shape {df.shape}")
        return df
    except Exception as e:  
        logger.error(f"Error loading data from {path}: {e}")
        raise


def evaluate(model, X, y_true, label_encoder):
    try:
        dtest = xgb.DMatrix(X)
        y_pred = model.predict(dtest)
        y_pred = (y_pred > 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        y_true_labels = label_encoder.inverse_transform(y_true)
        y_pred_labels = label_encoder.inverse_transform(y_pred)

        report = classification_report(y_true_labels, y_pred_labels, output_dict=True)

        logger.info(f"Evaluation results: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'classification_report': report
        }
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def save_metrics(metrics, path='reports/metrics.json'):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:  
            json.dump(metrics, f, indent=4)
        logger.debug(f"Metrics saved to {path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {path}: {e}")
        raise


def main():
    try:
        model_path = './models/xgb_model_fold_1.pkl'  
        label_encoder_path = './models/label_encoder.pkl'
        test_data_path = './data/processed/train_features.csv'

        # Load model and encoder
        model = load_pickle(model_path)
        with open(label_encoder_path, 'rb') as f:
            le = pickle.load(f)
        logger.debug(f"Label encoder loaded from {label_encoder_path}")

        # Load and prepare data
        test_data = load_data(test_data_path)
        X = test_data.drop(columns=['personality'])
        y = le.transform(test_data['personality']) 

        # Evaluate
        metrics = evaluate(model, X, y, le)
        save_metrics(metrics)

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
