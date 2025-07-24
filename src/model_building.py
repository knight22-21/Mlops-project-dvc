import os
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from utils.common import load_params

# Load parameters
params = load_params()
model_cfg = params.get('model_building', {})

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": model_cfg.get("max_depth", 4),
    "eta": model_cfg.get("eta", 0.1),
    "subsample": model_cfg.get("subsample", 0.8),
    "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
    "random_state": model_cfg.get("random_state", 42)
}
N_SPLITS = model_cfg.get("n_splits", 5)
NUM_BOOST_ROUNDS = model_cfg.get("n_estimators", 500)
EARLY_STOPPING_ROUNDS = model_cfg.get("early_stopping_rounds", 20)

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, 'model_building.log'))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(path)
        logger.debug(f"Data loaded successfully from {path}. Shape: {df.shape}")
        return df
    except Exception as e:  
        logger.error(f"Error loading data from {path}: {e}")
        raise
    
def train_xgboost(X, y, X_test=None):
    """Train an XGBoost model with cross-validation."""
    logger.debug("Starting cross-validated training with XGBoost...")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=XGB_PARAMS["random_state"])
    oof_preds = np.zeros(len(X))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.debug(f"Fold {fold+1} started")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            evals=[(dval, 'validation')],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=50
        )
        
        models.append(model)
        oof_preds[val_idx] = model.predict(dval) > 0.5
        logger.debug(f"Fold {fold+1} completed. OOF predictions updated.")
        
    acc = accuracy_score(y, oof_preds)
    logger.info(f"Cross-validated accuracy: {acc:.4f}")
    
    return models, oof_preds

def save_models(models, path: str):
    try:
        os.makedirs(path, exist_ok=True)
        for i, model in enumerate(models):
            model_path = os.path.join(path, f"xgb_model_fold_{i+1}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.debug(f"Model for fold {i+1} saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        raise

def main():
    try:
        df = load_data('./data/processed/train_features.csv')
        
        # Encode labels
        le = LabelEncoder()
        df['personality'] = le.fit_transform(df['personality'])
        
        X = df.drop(columns=['personality'])
        y = df['personality']
        
        models, oof_preds = train_xgboost(X, y)
        
        # save models
        save_models(models, path='models')
        
        # save label encoder
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
            logger.debug("Label encoder saved successfully.")   
        
        # Save out-of-fold predictions
        np.save('models/oof_preds.npy', oof_preds)
        logger.debug("Out-of-fold predictions saved successfully.") 
        
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
