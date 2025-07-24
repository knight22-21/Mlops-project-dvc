import os
import logging
import pandas as pd
from utils.common import load_params

# Load parameters
params = load_params()
preprocess_cfg = params['data_preprocessing']

# Logging setup
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def preprocess_data(train_df, test_df):
    """Preprocesses the training and test datasets:
    - Combines datasets for uniform transformation
    - Standardizes column names
    - Binary encodes yes/no columns
    - Fills missing values based on params
    """
    try:
        logger.debug("Copying original datasets")
        train = train_df.copy()
        test = test_df.copy()
        
        train['is_train'] = 1
        test['is_train'] = 0    
        test['personality'] = None
        
        df_all = pd.concat([train, test], ignore_index=True)
        
        logger.debug("Standardizing column names")
        df_all.columns = df_all.columns.str.lower().str.replace(' ', '_')
        
        logger.debug("Encoding binary columns")
        df_all['stage_fear'] = df_all['stage_fear'].map({'yes': 1, 'no': 0})
        df_all['drained_after_socializing'] = df_all['drained_after_socializing'].map({'yes': 1, 'no': 0})
        
        logger.debug("Handling missing values using parameters from params.yaml")
        cat_fill = preprocess_cfg.get('fillna_categorical', 'unknown')
        num_fill = preprocess_cfg.get('fillna_numeric', 'mean')

        cat_cols = df_all.select_dtypes(include=['object']).columns
        num_cols = df_all.select_dtypes(include=['number']).columns

        for col in cat_cols:
            df_all[col] = df_all[col].fillna(cat_fill)

        for col in num_cols:
            if num_fill == 'mean':
                df_all[col] = df_all[col].fillna(df_all[col].mean())
            elif num_fill == 'median':
                df_all[col] = df_all[col].fillna(df_all[col].median())
            elif num_fill == 'zero':
                df_all[col] = df_all[col].fillna(0)
            else:
                raise ValueError(f"Unsupported numeric fill strategy: {num_fill}")
        
        logger.debug("Preprocessing complete")
        train_cleaned = df_all[df_all['is_train'] == 1].drop(columns=['is_train'])
        test_cleaned = df_all[df_all['is_train'] == 0].drop(columns=['is_train', 'personality'])
        
        return train_cleaned, test_cleaned
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def main():
    try:
        logger.debug("Loading data")
        train = pd.read_csv('./data/raw/train.csv')
        test = pd.read_csv('./data/raw/test.csv')
        
        logger.debug("Starting preprocessing")
        train_cleaned, test_cleaned = preprocess_data(train, test)
        
        interim_path = './data/interim'
        os.makedirs(interim_path, exist_ok=True)
        train_cleaned.to_csv(os.path.join(interim_path, 'train_cleaned.csv'), index=False)
        test_cleaned.to_csv(os.path.join(interim_path, 'test_cleaned.csv'), index=False)
        logger.debug("Preprocessed data saved")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
