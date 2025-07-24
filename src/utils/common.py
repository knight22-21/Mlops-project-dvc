import yaml
import logging

def load_params(params_path: str = 'params.yaml') -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise
