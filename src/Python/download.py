import sys
sys.path.insert(0, 'src/Python/utils')
from utilities import (
    configure_logging, create_logger, stop_logger
)
from collect_data import download_data, combine_and_save_files, prepare_data

configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = ['nab']
)
logger = create_logger()

# NAB
# NOTE: nab data is downloaded https://github.com/numenta/NAB/tree/master
download_data('nab', save = True)

combine_and_save_files(
    path_list_to_read = ['data', 'nab'],
    path_list_to_write = ['data', 'nab'],
    name_list = ['nab']
)

prepare_data('nab', save = True)

stop_logger(logger)

# test
# import numpy as np
# from collect_data import get_data
# np.random.seed(1992)
# data = get_data(['data', 'nab'], ['nab', 'prep'], samples = 1)
# data
# data.is_real_anomaly.sum()