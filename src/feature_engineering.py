import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from utils.common import load_params

# Load parameters
params = load_params()
fe_cfg = params.get('feature_engineering', {})
unknown_val = fe_cfg.get('encode_unknown_value', -1)

# Logging setup
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def feature_engineering(train_df, test_df):
    """Perform feature engineering on the training and test datasets."""
    try:
        logger.debug("Copying original datasets.")
        train = train_df.copy()
        test = test_df.copy()
        
        train['is_train'] = 1
        test['is_train'] = 0   
        test['personality'] = np.nan
        
        logger.debug("Combining train and test datasets for feature engineering.")
        df_all = pd.concat([train, test], ignore_index=True)
        
        logger.debug("Creating new features.")
        df_all['event_ratio'] = df_all['social_event_attendance'] / (df_all['going_outside'] + 1)
        df_all['alone_social_ratio'] = df_all['time_spent_alone'] / (df_all['going_outside'] + 1)
        df_all['post_per_friend'] = df_all['post_frequency'] / (df_all['friends_circle_size'] + 1)
        df_all['social_energy'] = (
            df_all['social_event_attendance'] +
            df_all['going_outside'] +
            df_all['friends_circle_size'] +
            df_all['post_frequency'] -
            df_all['time_spent_alone'] -
            df_all['drained_after_socializing'] * 2
        )
        
        logger.debug("Encoding remaining categorical features using unknown_value from config.")
        cat_cols = df_all.select_dtypes(include='object').drop(columns=['personality'], errors='ignore').columns
        if len(cat_cols) > 0:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=unknown_val)
            df_all[cat_cols] = encoder.fit_transform(df_all[cat_cols])
        else:
            logger.debug("No categorical columns to encode.")
            
        logger.debug("Splitting combined dataset back into train and test sets.")
        train_feat = df_all[df_all['is_train'] == 1].drop(columns=['is_train'])
        test_feat = df_all[df_all['is_train'] == 0].drop(columns=['is_train', 'personality'])
        
        return train_feat, test_feat
    
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        print(f"Error: {e}")
    
def main():
    try:
        logger.debug("Loading cleaned data.")
        train = pd.read_csv('./data/interim/train_cleaned.csv')
        test = pd.read_csv('./data/interim/test_cleaned.csv')
        
        logger.debug("Starting feature engineering process.")
        train_fe, test_fe = feature_engineering(train, test)
        
        processed_path = './data/processed'
        os.makedirs(processed_path, exist_ok=True)
        
        train_fe.to_csv(os.path.join(processed_path, 'train_features.csv'), index=False)
        test_fe.to_csv(os.path.join(processed_path, 'test_features.csv'), index=False)
        
        logger.debug("Feature engineered data saved successfully.")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
