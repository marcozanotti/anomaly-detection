
import sys
sys.path.insert(0, 'src/Python/utils')
import gc
import time
import numpy as np
import pandas as pd
from utilities import save_data, get_dataset_frequency
from collect_data import get_data, combine_train_test
from set_engine import get_model_type, set_engine, add_data_features
from mlforecast.utils import PredictionIntervals

import logging
module_logger = logging.getLogger('fit_models')


def split_train_test(data, test_window):
    """
    Function to split the data into train and test dataframes.

    Args:
        data (pd.DataFrame): data in Nixtla's format.
        test_window (int or float): length of the test window.
            - If int: number of observations.
            - If float (0 < test_window < 1): proportion of observations.

    Returns:
        (pd.DataFrame, pd.DataFrame): training and test dataframes.
    """
    module_logger.info("Splitting data into train and test...")

    # Ensure data is ordered
    data = data.sort_values(["unique_id", "ds"])

    train_parts, test_parts = [], []

    for _, g in data.groupby("unique_id"):
        n = len(g)
        if isinstance(test_window, float) and 0 < test_window < 1:
            tw = max(1, int(n * test_window))  # at least 1 obs in test
        else:
            tw = int(test_window)

        train_g = g.iloc[:-tw] if tw > 0 else g.iloc[:]
        test_g = g.iloc[-tw:] if tw > 0 else g.iloc[0:0]

        train_parts.append(train_g)
        test_parts.append(test_g)

    train_df = pd.concat(train_parts).reset_index(drop = True)
    test_df = pd.concat(test_parts).reset_index(drop = True)

    module_logger.info(
        f"Train size: {len(train_df)} obs, Test size: {len(test_df)} obs"
    )

    return train_df, test_df

def get_retrain_ids(test_window, horizon, retrain_window = 1):

    """Function to get the retrain ids.

    Args:
        test_window (int): length of the test window.
        retrain_window (int): window for retraining. Defaults to 1.
    
    Returns:
        list: list of retrain ids.
    """
    res = list(range(0, (test_window - horizon + 1), retrain_window))
    return res

def get_model_name(engine):

    """Function to get the model name based on the engine.

    Args:
        engine (str): engine used for fitting the model.
    
    Returns:
        str: model name.
    """

    engine_class = str(engine.__class__)
    if engine_class == "<class 'statsforecast.core.StatsForecast'>":
        model_names = list(engine.models.keys())
    elif engine_class == "<class 'mlforecast.forecast.MLForecast'>":
        model_names = list(engine.models.keys())
    elif engine_class == "<class 'neuralforecast.core.NeuralForecast'>":
        model_names = engine.models
    else:
        raise ValueError(f'Invalid engine class {engine_class}.')

    if len(model_names) > 1:
        raise(f'Please specify only one model at a time')
    else:
        model_name = model_names[0]
        
    return model_name

def get_prediction_intervals(intervals):
    """Function to get prediction intervals.

    Args:
        intervals (fun): prediction intervals function.
    
    Returns:
        list: list of prediction intervals.
    """

    module_logger.info('Defining prediction intervals...')

    if intervals is not None:
        intervals_new = PredictionIntervals(
            h = intervals['h'],
            n_windows = intervals['n_windows'],
            method = intervals['method']
        )

    return intervals_new

def retrain_ml_model(
    train_df, 
    test_df,
    dataset_name,
    frequency,
    model_name,
    engine, 
    test_window,
    horizon, 
    retrain_window,
    features,
    intervals = None,
    levels = [50, 60, 70, 80, 90, 95, 99],
    store_in_sample_results = False,
    ext = '.parquet'
):

    """Function to retrain the ML model and predict with retrained model.

    Args:
        train_df (pd.DataFrame): training data in Nixtla's format.
        test_df (pd.DataFrame): testing data in Nixtla's format.
        dataset_name (str): name of the dataset (e.g., 'm5', 'm4').
        frequency (str): frequency of the data (e.g., 'daily', 'weekly').
        model_name (str): name of the model.
        engine (MLForecast class): ML model engine.
        test_window (int): length of the test window.
        horizon (int): forecasting horizon.
        retrain_window (int, optional): window for retraining.
        features (dict): features to be used for training.
        intervals (fun): intervals to be used for predictions. Defaults to None.
        levels (list): confidence levels for the predictions. Defaults to
        [60, 70, 80, 85, 90, 95, 99].
        store_in_sample_results (bool, optional): store in-sample results.
        Defaults to False.
        ext (str, optional): file extension for storing results. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: predictions made by the retrained models.
    """

    module_logger.info('---------------------------------------------------------------')
    start_time = time.time()

    # define the model name
    # model_name = get_model_name(engine)
    module_logger.info(f'[ Model: {model_name} | Retrain Window: {retrain_window} ]')

    # define the fitting times
    fitting_ids = get_retrain_ids(test_window, horizon, retrain_window)
    n_fitting = len(fitting_ids) # int(np.round(test_window / retrain_window, 0))
    n_samples = test_window - horizon + 1
    module_logger.info(
        f'[ Retrain ids: {fitting_ids} ] | Num fitting: {n_fitting} | Num iterations: {n_samples} ]'
    )

    # define static features
    static_features = features['static']

    module_logger.info(f'Train dataset contains: {list(train_df.columns)}...')
    module_logger.info(f'Test dataset contains: {list(test_df.columns)}...')

    # initialize the time dataframe (the only auto-incremental df with save at the end)
    time_df = pd.DataFrame()

    for i in range(n_samples):
        
        module_logger.info(f'Step {i + 1} of {n_samples}')

        # define the training data
        train_df_tmp = combine_train_test(train_df, test_df.groupby('unique_id').head(i))

        # define the testing data
        # test_df_tmp = test_df.groupby('unique_id').head(i + horizon)
        # test_df_tmp.reset_index(drop = True, inplace = True)
        # ds_to_remove = train_df_tmp["ds"].unique()
        # test_df_tmp = test_df_tmp.loc[~test_df_tmp['ds'].isin(ds_to_remove)]
        test_df_tmp = test_df.groupby('unique_id').head(i + horizon).groupby('unique_id').tail(horizon)
        test_df_tmp.reset_index(drop = True, inplace = True)
        # remove static features because they are used in fitting only
        test_df_tmp.drop(columns = static_features, axis = 1, inplace = True)
        
        if i in fitting_ids:

            # re-train the model
            module_logger.info(f'Fitting: t = {i}, {int(i / retrain_window + 1)} of {n_fitting}...')
            start_fit_time = time.time()

            if intervals is None:
                fit_tmp = engine.fit(
                    df = train_df_tmp, 
                    static_features = static_features,
                    fitted = store_in_sample_results
                )
            else:
                fit_tmp = engine.fit(
                    df = train_df_tmp, 
                    static_features = static_features,
                    fitted = store_in_sample_results,
                    prediction_intervals = intervals
                )

            end_fit_time = time.time()

            # predict out-of-sample with the models
            module_logger.info('Predicting...')
            start_predict_time = time.time()
            out_sample_df_tmp = fit_tmp.predict(h = horizon, level = levels, X_df = test_df_tmp)
            end_predict_time = time.time()

            tot_sample_time = end_predict_time - start_fit_time

            if store_in_sample_results:

                # extract in-sample results from the model only when fitting
                module_logger.info('Extracting fitted values...')
                in_sample_df_tmp = fit_tmp.fcst_fitted_values_
                # add additional columns to the dataframes for tracking parameters
                in_sample_df_tmp['sample'] = i
                in_sample_df_tmp['method'] = model_name
                in_sample_df_tmp['test_window'] = test_window
                in_sample_df_tmp['horizon'] = horizon
                in_sample_df_tmp['retrain_window'] = retrain_window
                # format column names and reset index values
                in_sample_df_tmp = in_sample_df_tmp.rename(columns = {model_name: 'fit'})
                in_sample_df_tmp.reset_index(drop = True, inplace = True)
                # save to file
                save_data(
                    data = in_sample_df_tmp, 
                    path_list = ['results', dataset_name, frequency, model_name, retrain_window, 'insample', 'tmp'],
                    name_list = [dataset_name, frequency, model_name, retrain_window, 'insample', i],
                    ext = ext
                )
                del in_sample_df_tmp
                
        else:

            # update the mlforecast object with the new data 
            # NOTE: fundamental to roll predictions without fitting !!!!!
            engine.update(train_df_tmp.groupby('unique_id').tail(1))

            module_logger.info('Predicting with pre-trained model...')
            start_predict_time = time.time()
            out_sample_df_tmp = fit_tmp.predict(h = horizon, level = levels, X_df = test_df_tmp)
            end_predict_time = time.time()

            tot_sample_time = end_predict_time - start_predict_time
        
        # add additional columns to the dataframes for tracking parameters
        out_sample_df_tmp['sample'] = i
        out_sample_df_tmp['method'] = model_name
        out_sample_df_tmp['test_window'] = test_window
        out_sample_df_tmp['horizon'] = horizon
        out_sample_df_tmp['retrain_window'] = retrain_window
        # add actual out-of-sample to results
        out_sample_df_tmp = out_sample_df_tmp.merge(
            test_df_tmp[['unique_id', 'ds', 'y']], 
            how = 'left', 
            on = ['unique_id', 'ds'], 
            copy = False
        )
        # format column names and reset index values
        out_sample_df_tmp.columns = out_sample_df_tmp.columns.str.replace(model_name, 'fcst')
        out_sample_df_tmp.reset_index(drop = True, inplace = True)
        # save to file
        save_data(
            data = out_sample_df_tmp, 
            path_list = ['results', dataset_name, frequency, model_name, retrain_window, 'outsample', 'tmp'],
            name_list = [dataset_name, frequency, model_name, retrain_window, 'outsample', i],
            ext = ext
        ) 
        
        # store computing time information for each sample
        time_df_tmp = pd.DataFrame({
            'sample': i,
            'total_fit_time': [end_fit_time - start_fit_time],
            'total_predict_time': [end_predict_time - start_predict_time],
            'total_sample_time': tot_sample_time
        })
        time_df = pd.concat([time_df, time_df_tmp], axis = 0)

        del train_df_tmp, test_df_tmp, out_sample_df_tmp, time_df_tmp # ds_to_remove 
        if (i % 10) == 0:
            gc.collect() # call gc once every 10 iterations to avoid overhead

    # add additional columns to the dataframes for tracking parameters
    time_df['method'] = model_name
    time_df['test_window'] = test_window
    time_df['horizon'] = horizon
    time_df['retrain_window'] = retrain_window
    time_df.reset_index(drop = True, inplace = True)
    # save to file
    save_data(
        data = time_df, 
        path_list = ['results', dataset_name, frequency, model_name, 'time', 'byretrain'],
        name_list = [dataset_name, frequency, model_name, retrain_window, 'time'],
        ext = ext
    )

    end_time = time.time()
    tot_time = end_time - start_time
    module_logger.info(f'Total computing time: {tot_time:.1f} seconds')

    return

def retrain_dl_model(
    train_df, 
    test_df,
    dataset_name,
    frequency,
    model_name,
    engine, 
    test_window,
    horizon, 
    retrain_window,
    features,
    intervals = None,
    levels = [50, 60, 70, 80, 90, 95, 99],
    store_in_sample_results = False,
    ext = '.parquet'
):

    """Function to retrain the DL model and predict with retrained model.

    Args:
        train_df (pd.DataFrame): training data in Nixtla's format.
        test_df (pd.DataFrame): testing data in Nixtla's format.
        dataset_name (str): name of the dataset (e.g., 'm5', 'm4').
        frequency (str): frequency of the data (e.g., 'daily', 'weekly').
        model_name (str): name of the model.
        engine (NeuralForecast class): DL model engine.
        test_window (int): length of the test window.
        horizon (int): forecasting horizon.
        retrain_window (int, optional): window for retraining.
        features (dict): features to be used for training.
        intervals (fun): intervals to be used for predictions. Defaults to None.
        levels (list): confidence levels for the predictions. Defaults to
        [60, 70, 80, 85, 90, 95, 99].
        store_in_sample_results (bool, optional): store in-sample results.
        Defaults to False.
        ext (str, optional): file extension for storing results. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: predictions made by the retrained models.
    """

    module_logger.info('---------------------------------------------------------------')
    start_time = time.time()

    # define the model name
    # model_name = get_model_name(engine)
    module_logger.info(f'[ Model: {model_name} | Retrain Window: {retrain_window} ]')

    # define the fitting times
    fitting_ids = get_retrain_ids(test_window, horizon, retrain_window)
    n_fitting = len(fitting_ids) # int(np.round(test_window / retrain_window, 0))
    n_samples = test_window - horizon + 1
    module_logger.info(
        f'[ Retrain ids: {fitting_ids} ] | Num fitting: {n_fitting} | Num iterations: {n_samples} ]'
    )

    # define the static features
    static_features = features['static']

    # get the static dataframe and remove it from train and test
    static_df = test_df[['unique_id'] + static_features].drop_duplicates().reset_index(drop = True)
    train_df = add_data_features(data = train_df, frequency = get_dataset_frequency(dataset_name), features = features, remove_static = True)
    test_df = add_data_features(data = test_df, frequency = get_dataset_frequency(dataset_name), features = features, remove_static = True)
    
    module_logger.info(f'Train dataset contains: {list(train_df.columns)}...')
    module_logger.info(f'Test dataset contains: {list(test_df.columns)}...')

    # initialize the time dataframe (the only auto-incremental df with save at the end)
    time_df = pd.DataFrame()

    for i in range(n_samples):
        
        module_logger.info(f'Step {i + 1} of {n_samples}')

        # define the training data
        train_df_tmp = combine_train_test(train_df, test_df.groupby('unique_id').head(i))

        # define the testing data
        # test_df_tmp = test_df.groupby('unique_id').head(i + horizon)
        # test_df_tmp.reset_index(drop = True, inplace = True)
        # ds_to_remove = train_df_tmp["ds"].unique()
        # test_df_tmp = test_df_tmp.loc[~test_df_tmp['ds'].isin(ds_to_remove)]
        test_df_tmp = test_df.groupby('unique_id').head(i + horizon).groupby('unique_id').tail(horizon)
        test_df_tmp.reset_index(drop = True, inplace = True)
        
        if i in fitting_ids:

            # re-train the model
            module_logger.info(f'Fitting: t = {i}, {int(i / retrain_window + 1)} of {n_fitting}...')
            start_fit_time = time.time()

            if intervals is None:
                engine.fit(df = train_df_tmp, static_df = static_df)
            else:
                engine.fit(df = train_df_tmp, static_df = static_df, prediction_intervals = intervals)

            end_fit_time = time.time()

            # predict out-of-sample with the models
            module_logger.info('Predicting...')
            start_predict_time = time.time()
            out_sample_df_tmp = engine.predict(
                futr_df = test_df_tmp.drop(columns = 'y', axis = 1), 
                level = levels
            )
            end_predict_time = time.time()

            tot_sample_time = end_predict_time - start_fit_time

            if store_in_sample_results:

                # extract in-sample results from the model only when fitting
                module_logger.info('Extracting fitted values...')
                in_sample_df_tmp = engine.predict_insample() # not working since time series with different lengths
                in_sample_df_tmp.drop(columns = 'cutoff', inplace = True)
                # add additional columns to the dataframes for tracking parameters
                in_sample_df_tmp['sample'] = i
                in_sample_df_tmp['method'] = model_name
                in_sample_df_tmp['test_window'] = test_window
                in_sample_df_tmp['horizon'] = horizon
                in_sample_df_tmp['retrain_window'] = retrain_window
                # format column names and reset index values
                in_sample_df_tmp = in_sample_df_tmp.rename(columns = {model_name: 'fit'})
                in_sample_df_tmp.reset_index(drop = True, inplace = True)
                # save to file
                save_data(
                    data = in_sample_df_tmp, 
                    path_list = ['results', dataset_name, frequency, model_name, retrain_window, 'insample', 'tmp'],
                    name_list = [dataset_name, frequency, model_name, retrain_window, 'insample', i],
                    ext = ext
                )
                del in_sample_df_tmp
                
        else:

            module_logger.info('Predicting with pre-trained model...')
            start_predict_time = time.time()
            out_sample_df_tmp = engine.predict(
                df = train_df_tmp,
                static_df = static_df,
                futr_df = test_df_tmp.drop(columns = 'y', axis = 1), 
                level = levels
            )
            end_predict_time = time.time()

            tot_sample_time = end_predict_time - start_predict_time
        
        # add additional columns to the dataframes for tracking parameters
        out_sample_df_tmp['sample'] = i
        out_sample_df_tmp['method'] = model_name
        out_sample_df_tmp['test_window'] = test_window
        out_sample_df_tmp['horizon'] = horizon
        out_sample_df_tmp['retrain_window'] = retrain_window
        # add actual out-of-sample to results
        out_sample_df_tmp = out_sample_df_tmp.merge(
            test_df_tmp[['unique_id', 'ds', 'y']], 
            how = 'left', 
            on = ['unique_id', 'ds'], 
            copy = False
        )
        # format column names and reset index values
        out_sample_df_tmp.columns = out_sample_df_tmp.columns.str.replace(model_name, 'fcst')
        out_sample_df_tmp.reset_index(drop = True, inplace = True)
        # save to file
        save_data(
            data = out_sample_df_tmp, 
            path_list = ['results', dataset_name, frequency, model_name, retrain_window, 'outsample', 'tmp'],
            name_list = [dataset_name, frequency, model_name, retrain_window, 'outsample', i],
            ext = ext
        ) 
        
        # store computing time information for each sample
        time_df_tmp = pd.DataFrame({
            'sample': i,
            'total_fit_time': [end_fit_time - start_fit_time],
            'total_predict_time': [end_predict_time - start_predict_time],
            'total_sample_time': tot_sample_time
        })
        time_df = pd.concat([time_df, time_df_tmp], axis = 0)

        del train_df_tmp, test_df_tmp, out_sample_df_tmp, time_df_tmp # ds_to_remove
        if (i % 10) == 0:
            gc.collect() # call gc once every 10 iterations to avoid overhead

    # add additional columns to the dataframes for tracking parameters
    time_df['method'] = model_name
    time_df['test_window'] = test_window
    time_df['horizon'] = horizon
    time_df['retrain_window'] = retrain_window
    time_df.reset_index(drop = True, inplace = True)
    # save to file
    save_data(
        data = time_df, 
        path_list = ['results', dataset_name, frequency, model_name, 'time', 'byretrain'],
        name_list = [dataset_name, frequency, model_name, retrain_window, 'time'],
        ext = ext
    )

    end_time = time.time()
    tot_time = end_time - start_time
    module_logger.info(f'Total computing time: {tot_time:.1f} seconds')

    return

def retrain_model(config):

    """Function to retrain the models and predict with retrained models
    for different retraining scenarios.

    Args:
        config (dict): Configuration parameters.

    Returns:
        pd.DataFrame: predictions made by the retrained models.
    """

    module_logger.info('===============================================================')

    # dataset parameters
    dataset_name = config['dataset']['dataset_name']
    frequency = config['dataset']['frequency']
    min_series_length = config['dataset']['min_series_length']
    max_series_length = config['dataset']['max_series_length']
    samples = config['dataset']['samples']
    ext = config['dataset']['ext']
    seed = config['dataset']['seed']
    # fitting parameters
    test_window = config['fitting']['test_window']
    horizon = config['fitting']['horizon']
    retrain_scenarios = config['fitting']['retrain_scenarios']
    intervals = config['fitting']['intervals']
    levels = config['fitting']['levels']
    store_in_sample_results = config['fitting']['store_in_sample_results']
    # model parameters
    model_names = config['model_names']
    model_params = config['model_params']
    target_transforms = config['target_transforms']
    features = config['features']

    # load the dataset
    if samples is not None:
        np.random.seed(seed)
    data = get_data(
        path_list = ['data', dataset_name],
        name_list = [dataset_name, frequency, 'prep'],
        ext = '.parquet',
        min_series_length = min_series_length,
        max_series_length = max_series_length,
        samples = samples
    )
    # split the data into train and test dataframes
    train_df, test_df = split_train_test(data, test_window)
    del data

    # define prediction intervals
    intervals = get_prediction_intervals(intervals)

    for m in model_names:

        module_logger.info('---------------------------- START ----------------------------')
        
        model_type = get_model_type(m)
        module_logger.info(f'[ Model type: {model_type} | Model name: {m} ]')
        
        if model_params is None:
            engine_tmp = set_engine(m, get_dataset_frequency(dataset_name), features, target_transforms, model_params)
        else:
            engine_tmp = set_engine(m, get_dataset_frequency(dataset_name), features, target_transforms, model_params[m])

        for rs in retrain_scenarios:

            if model_type == 'sf':

                # retrain_sf_model(
                #     train_df = train_df, 
                #     test_df = test_df,
                #     dataset_name = dataset_name,
                #     frequency = frequency,
                #     model_name = m,
                #     engine = engine_tmp,
                #     test_window = test_window,
                #     horizon = horizon,
                #     retrain_window = rs,
                #     store_in_sample_results = store_in_sample_results,
                #     ext = ext
                # )
                raise ValueError('Not yet implemented.')

            elif model_type == 'ml':

                retrain_ml_model(
                    train_df = train_df, 
                    test_df = test_df,
                    dataset_name = dataset_name,
                    frequency = frequency,
                    model_name = m,
                    engine = engine_tmp,
                    test_window = test_window,
                    horizon = horizon,
                    retrain_window = rs,
                    features = features,
                    intervals = intervals,
                    levels = levels,
                    store_in_sample_results = store_in_sample_results,
                    ext = ext
                )

            elif model_type == 'dl':

                retrain_dl_model(
                    train_df = train_df, 
                    test_df = test_df,
                    dataset_name = dataset_name,
                    frequency = frequency,
                    model_name = m,
                    engine = engine_tmp,
                    test_window = test_window,
                    horizon = horizon,
                    retrain_window = rs,
                    features = features,
                    intervals = intervals,
                    levels = levels,
                    store_in_sample_results = store_in_sample_results,
                    ext = ext
                )    
            
            else:
                raise ValueError('Not yet implemented.')
        
        module_logger.info('----------------------------- END -----------------------------')
    
    module_logger.info('===============================================================')

    return
