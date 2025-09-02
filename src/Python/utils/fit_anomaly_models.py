
import sys
sys.path.insert(0, 'src/Python/utils')
import gc
import time
import numpy as np
import pandas as pd
from utilities import save_data
from collect_data import get_data, combine_train_test
from set_anomaly_engine import get_anomaly_model_type, set_anomaly_engine
from fit_models import split_train_test, get_retrain_ids

import logging
module_logger = logging.getLogger('fit_anomaly_models')

pd.options.mode.copy_on_write = True

def retrain_model(
    train_df, 
    test_df,
    dataset_name,
    model_name,
    engine, 
    test_window,
    horizon, 
    retrain_window,
    # store_in_sample_results = False,
    ext = '.parquet'
):

    """Function to train the Statistic anomaly detection model and predict with trained model.

    Args:
        train_df (pd.DataFrame): training data in Nixtla's format.
        test_df (pd.DataFrame): testing data in Nixtla's format.
        dataset_name (str): name of the dataset (e.g., 'nab').
        model_name (str): name of the model.
        engine: Anomaly model engine.
        test_window (int): length of the test window.
        horizon (int): forecasting horizon.
        retrain_window (int, optional): window for retraining.
        ext (str, optional): file extension for storing results. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: predictions made by the retrained models.
    """

    module_logger.info('---------------------------------------------------------------')
    start_time = time.time()

    # define the model name
    # model_name = get_model_name(engine)
    module_logger.info(f'[ Model: {model_name} | Retrain Window: {retrain_window} ]')
    module_logger.info(f'Train dataset contains: {list(train_df.columns)}...')
    module_logger.info(f'Test dataset contains: {list(test_df.columns)}...')

    # define the fitting times
    series = list(train_df['unique_id'].unique())
    n_series = len(series)
    module_logger.info(f'[ Num series: {n_series} ]')

    out_sample_df = pd.DataFrame() # initialize the predictions dataframe
    time_df = pd.DataFrame() # initialize the time dataframe

    for ts in series:

        module_logger.info(f'Series {ts}')
        train_df_ts = train_df.query(f'unique_id == "{ts}"') 
        test_df_ts = test_df.query(f'unique_id == "{ts}"')

        if isinstance(test_window, float):
            test_window_ts = len(test_df_ts)
        else:
            test_window_ts = test_window
        fitting_ids = get_retrain_ids(test_window_ts, horizon, retrain_window)
        n_fitting = len(fitting_ids) # int(np.round(test_window / retrain_window, 0))
        n_samples = test_window_ts - horizon + 1
        module_logger.info(
            f'[ Retrain ids: {fitting_ids} ] | Num fitting: {n_fitting} | Num iterations: {n_samples} ]'
        )

        for i in range(n_samples):
            
            module_logger.info(f'Step {i + 1} of {n_samples}')

            # define the training data
            train_df_ts_tmp = combine_train_test(train_df_ts, test_df_ts.groupby('unique_id').head(i + horizon))

            # define the testing data
            test_df_ts_tmp = test_df_ts.groupby('unique_id').head(i + horizon).groupby('unique_id').tail(horizon)
            test_df_ts_tmp.reset_index(drop = True, inplace = True)
                
            # re-train the model
            module_logger.info(f'Fitting: t = {i}, {int(i / retrain_window + 1)} of {n_fitting}...')
            start_fit_time = time.time()
            fit_ts_tmp = engine(df = train_df_ts_tmp)
            end_fit_time = time.time()

            # predict out-of-sample with the models
            module_logger.info('Predicting...')
            start_predict_time = time.time()
            out_sample_df_ts_tmp = fit_ts_tmp.tail(horizon)
            end_predict_time = time.time()

            tot_sample_time = end_predict_time - start_fit_time
                            
            # add additional columns to the dataframes for tracking parameters
            out_sample_df_ts_tmp['sample'] = i
            # format column names and reset index values
            out_sample_df_ts_tmp.columns = out_sample_df_ts_tmp.columns.str.replace(model_name, 'anomaly')
            out_sample_df_ts_tmp.reset_index(drop = True, inplace = True)
            out_sample_df = pd.concat([out_sample_df, out_sample_df_ts_tmp], axis = 0)

            # store computing time information for each sample
            time_df_tmp = pd.DataFrame({
                'unique_id': ts,
                'sample': i,
                'total_fit_time': [end_fit_time - start_fit_time],
                'total_predict_time': [end_predict_time - start_predict_time],
                'total_sample_time': tot_sample_time
            })
            time_df = pd.concat([time_df, time_df_tmp], axis = 0)

            del train_df_ts_tmp, test_df_ts_tmp, fit_ts_tmp, out_sample_df_ts_tmp, time_df_tmp 
            if (i % 10) == 0:
                gc.collect() # call gc once every 10 iterations to avoid overhead

        del train_df_ts, test_df_ts, test_window_ts, fitting_ids, n_fitting, n_samples     

    # add information to predictions dataframe
    out_sample_df['method'] = model_name
    out_sample_df['test_window'] = test_window
    out_sample_df['horizon'] = horizon
    out_sample_df['retrain_window'] = retrain_window 
    out_sample_df.reset_index(drop = True, inplace = True)
    # save to file
    save_data(
        data = out_sample_df, 
        path_list = ['results', dataset_name, model_name, retrain_window, 'outsample', 'tmp'],
        name_list = [dataset_name, model_name, retrain_window, 'outsample'],
        ext = ext
    ) 
                    
    # add information to time dataframe
    time_df['method'] = model_name
    time_df['test_window'] = test_window
    time_df['horizon'] = horizon
    time_df['retrain_window'] = retrain_window
    time_df.reset_index(drop = True, inplace = True)
    # save to file
    save_data(
        data = time_df, 
        path_list = ['results', dataset_name, model_name, 'time', 'byretrain'],
        name_list = [dataset_name, model_name, retrain_window, 'time'],
        ext = ext
    )

    end_time = time.time()
    tot_time = end_time - start_time
    module_logger.info(f'Total computing time: {tot_time:.1f} seconds')

    return

def retrain_anomaly_model(config):

    """Function to train the anomaly models and predict with trained models.

    Args:
        config (dict): Configuration parameters.

    Returns:
        pd.DataFrame: predictions made by the trained models.
    """

    module_logger.info('===============================================================')

    # dataset parameters
    dataset_name = config['dataset']['dataset_name']
    min_series_length = config['dataset']['min_series_length']
    max_series_length = config['dataset']['max_series_length']
    samples = config['dataset']['samples']
    ext = config['dataset']['ext']
    seed = config['dataset']['seed']
    # fitting parameters
    test_window = config['fitting']['test_window']
    horizon = config['fitting']['horizon']
    retrain_scenarios = config['fitting']['retrain_scenarios']
    # model parameters
    model_names = config['model_names']
    model_params = config['model_params']

    # load the dataset
    if samples is not None:
        np.random.seed(seed)
    data = get_data(
        path_list = ['data', dataset_name],
        name_list = [dataset_name, 'prep'],
        ext = '.parquet',
        min_series_length = min_series_length,
        max_series_length = max_series_length,
        samples = samples
    )
    # split the data into train and test dataframes
    train_df, test_df = split_train_test(data, test_window)
    del data

    for m in model_names:

        module_logger.info('---------------------------- START ----------------------------')
        
        model_type = get_anomaly_model_type(m)
        module_logger.info(f'[ Model type: {model_type} | Model name: {m} ]')
        
        if model_params is None:
            engine_tmp = set_anomaly_engine(m, model_params)
        else:
            engine_tmp = set_anomaly_engine(m, model_params[m])

        for rs in retrain_scenarios:

            retrain_model(
                train_df = train_df, 
                test_df = test_df,
                dataset_name = dataset_name,
                model_name = m,
                engine = engine_tmp,
                test_window = test_window,
                horizon = horizon,
                retrain_window = rs,
                # store_in_sample_results = store_in_sample_results,
                ext = ext
            )    
        
        module_logger.info('----------------------------- END -----------------------------')
    
    module_logger.info('===============================================================')

    return
