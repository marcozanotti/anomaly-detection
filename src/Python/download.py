import pandas as pd
from src.Python.utils.collect_data import download_data, prepare_data

pd.set_option("display.max_rows", 4)



# M5
# NOTE: m5 data is downloaded from Nixtla's benchmark
download_data('m5', save = True)

# daily
prepare_data('m5', 'daily', static_features = True, xregs = True, save = True)
# weekly
# prepare_data('m5', 'weekly', static_features = True, xregs = True, save = True)
# monthly
# prepare_data('m5', 'monthly', static_features = True, xregs = True, save = True)



# VN1
# NOTE: vn1 data must be downloaded from Datasource.ai
# https://www.datasource.ai/competitions/phase-2-vn1-forecasting-accuracy-challenge/
# and saved into the data/vn1/ directory before proceeding
download_data('vn1', save = True)

# weekly
prepare_data('vn1', 'weekly', static_features = True, xregs = False, save = True)
# monthly
# prepare_data('vn1', 'monthly', static_features = True, xregs = False, save = True)


# M4 (Daily)
# NOTE: m4 data must be downloaded from Kaggle
# https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset
# and saved into the data/m4/ directory before proceeding
download_data('m4', save = True)

# daily
prepare_data('m4', 'daily', static_features = True, xregs = False, save = True)



# checks
# from src.Python.utils.collect_data import get_data
# from src.Python.utils.utilities import configure_logging, create_logger, stop_logger

# dataset_name = 'm5'
# frequency = 'daily'

# configure_logging(
#     config_file = 'config/log_config.yaml', 
#     name_list = [dataset_name, frequency, 'download']
# )
# logger = create_logger()


# data = get_data(
#     path_list = ['data', dataset_name], 
#     name_list = [dataset_name, frequency, 'prep'],
#     ext = '.parquet'
# )
# len(data['unique_id'].unique())
# data['ds'].min()
# data['ds'].max()
# data.groupby('unique_id')['ds'].min()
# data.groupby('unique_id')['ds'].max()

# get_data(
#     path_list = ['data', dataset_name],  
#     name_list = [dataset_name, frequency, 'prep'],
#     ext = '.parquet', 
#     min_series_length = 12 * 3
# )

# stop_logger(logger)

