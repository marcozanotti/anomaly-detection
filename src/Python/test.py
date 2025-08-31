import sys
sys.path.insert(0, 'src/Python/utils')
from utilities import (
    configure_logging, create_logger, stop_logger
)
from collect_data import get_data
from anomaly_detection import zscore


configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = ['nab']
)
logger = create_logger()


data = get_data(['data', 'nab'], ['nab', 'prep'])
data = data.query('unique_id == "speed_7578"')

zscore(data, q = 3)

stop_logger(logger)






import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from fit_anomaly_models import train_anomaly_model

config = get_config('config/TEST_train_Zscore_nab_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        'train', 'anomaly'
    ]
)
logger = create_logger()

train_anomaly_model(config = config)

stop_logger(logger)

