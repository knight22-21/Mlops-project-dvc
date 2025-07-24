import pandas as pd
import os
import logging
import yaml

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

logger= logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

#File handler
fh = logging.FileHandler(os.path.join(LOG_DIR, 'data_ingestion.log'))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

# YAML Params Loader
# def load_params(params_path: str = 'params.yaml') -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logger.info(f"Parameters loaded successfully from {params_path}")
#         return params
#     except Exception as e:
#         logger.error(f" Failed to load parameters: {e}")
#         raise

# Data Loader
def load_data(train_url: str, test_url: str, submission_url: str):
    """Load train, test, and submission datasets from given URLs."""
    try:
        train_df = pd.read_csv(train_url)
        test_df = pd.read_csv(test_url)
        submission_df = pd.read_csv(submission_url)
        
        logger.debug("Data loaded successfully from urls.")
        return train_df, test_df, submission_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    
# Data Saver
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, submission_df: pd.DataFrame, output_dir: str = './data/raw'):
    """Save train, test, and submission datasets to the specified directory."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        submission_df.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
        logger.debug("Train, test, and submission files saved to {output_dir.")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise
    
# Main
def main():
    try:
        # params = load_params()
        # ingestion_cfg = params['data_ingestion']

        # Provide your own dataset URLs here
        # train_url = ingestion_cfg['train_url']      # <--- You provide this
        # test_url = ingestion_cfg['test_url']        # <--- You provide this
        # submission_url = ingestion_cfg['submission_url']  # <--- You provide this

        train_url = 'https://raw.githubusercontent.com/knight22-21/Dataset/refs/heads/main/train.csv'  
        test_url = 'https://raw.githubusercontent.com/knight22-21/Dataset/refs/heads/main/test.csv'    
        submission_url = 'https://raw.githubusercontent.com/knight22-21/Dataset/refs/heads/main/sample_submission.csv'  

        train_df, test_df, submission_df = load_data(train_url, test_url, submission_url)
        save_data(train_df, test_df, submission_df, output_dir= './data/raw')

    except Exception as e:
        logger.error(f" Data ingestion failed: {e}")
        print(f"Error in data ingestion: {e}")

if __name__ == '__main__':
    main()