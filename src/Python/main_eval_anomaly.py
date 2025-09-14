import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from evaluate_anomaly import evaluate_anomaly_model

os.environ['NIXTLA_ID_AS_COL'] = '1'

config = get_config('config/eval/eval_nab_anomaly_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        'anomaly', 'evaluation'
    ]
)
logger = create_logger()

evaluate_anomaly_model(config = config)

stop_logger(logger)
