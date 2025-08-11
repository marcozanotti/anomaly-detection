
import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from fit_models import retrain_model

os.environ['NIXTLA_ID_AS_COL'] = '1'
# config = get_config('config/fit/TEST_retrain_LR_m5_daily_config.yaml')
# config = get_config('config/fit/TEST_retrain_LR_vn1_weekly_config.yaml')
# config = get_config('config/fit/retrain_ml_vn1_monthly_config.yaml')
# config = get_config('config/fit/retrain_ml_vn1_weekly_config.yaml')
# config = get_config('config/fit/retrain_ml_m5_monthly_config.yaml')
# config = get_config('config/fit/retrain_ml_m5_weekly_config.yaml')
# config = get_config('config/fit/retrain_ml_m5_daily_config.yaml')
config = get_config('config/fit/retrain_ml_m4_daily_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        config['dataset']['frequency'], 
        'retrain', 'ml'
    ]
)
logger = create_logger()

retrain_model(config = config)

stop_logger(logger)
