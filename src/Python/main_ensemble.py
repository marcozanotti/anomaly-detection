import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from fit_ensembles import fit_ensembles

os.environ['NIXTLA_ID_AS_COL'] = '1'
# config = get_config('config/ensemble/ensemble_vn1_monthly_config.yaml')
# config = get_config('config/ensemble/ensemble_vn1_weekly_config.yaml')
# config = get_config('config/ensemble/ensemble_m5_monthly_config.yaml')
# config = get_config('config/ensemble/ensemble_m5_weekly_config.yaml')
config = get_config('config/ensemble/ensemble_m4_daily_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'ensemble'
    ]
)
logger = create_logger()

fit_ensembles(config = config)

stop_logger(logger)
