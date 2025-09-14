import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from fit_ensembles import fit_ensembles

os.environ['NIXTLA_ID_AS_COL'] = '1'

config = get_config('config/ensemble/ensemble_nab_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        'ensemble'
    ]
)
logger = create_logger()

fit_ensembles(config = config)

stop_logger(logger)
