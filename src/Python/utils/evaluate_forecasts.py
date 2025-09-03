
import sys
sys.path.insert(0, 'src/Python/utils')
import gc
import numpy as np
import pandas as pd
import pandas_flavor as pf
from functools import partial
from utilsforecast.losses import (
    bias, mae, mse, rmse, mape, smape, mase, msse, rmsse, 
    quantile_loss, mqloss, scaled_quantile_loss, scaled_mqloss,
    coverage, calibration, scaled_crps
)
from utilsforecast.evaluation import evaluate
from utilities import (
    create_file_path, create_file_name, get_file_name, 
    save_data, load_data, combine_and_save_files, get_frequency
)
from collect_data import get_data
from fit_models import get_retrain_ids

import logging
module_logger = logging.getLogger('evaluate_forecasts')

def get_aggregate_function(function_name):
    """Function to get the aggregate function.

    Args:
        function_name (str): name of the function.
    
    Returns:
        function: aggregate function.
    """
    
    module_logger.info('Defining aggregate function...')

    if function_name == 'mean':
        return np.mean
    elif function_name == 'median':
        return np.median
    elif function_name == 'std':
        return np.std
    elif function_name == 'max':
        return np.max
    elif function_name =='min':
        return np.min
    elif function_name == 'sum':
        return np.sum 
    else:
        raise ValueError(f'Invalid aggregate function {function_name}')

@pf.register_dataframe_method
def aggregate_data(
    data, 
    group_columns, 
    drop_columns = None, 
    function_name = 'mean', 
    adjust_metrics = False
):

    """Function to aggregate evaluation metrics.

    Args:
        data (pd.DataFrame): dataframe to aggregate.
        group_columns (list): list of columns to group by.
        function_name (str, optional): function to use to aggregate. 
        Defaults to 'mean'.
        adjust_metrics (bool, optional): whether to adjust metrics. Defaults to False.
    
    Returns:
        pd.DataFrame: dataframe with aggregated data.
    """

    data_agg = data.copy()

    module_logger.info('Aggregating data...')
    if drop_columns is not None:
        data_agg.drop(columns = drop_columns, inplace = True)

    data_agg = data_agg \
        .groupby(group_columns) \
        .agg(function_name) \
        .reset_index()
    
    if adjust_metrics:

        if 'mse' in data_agg.columns:
            data_agg['rmse'] = np.sqrt(data_agg['mse'])

        if 'msse' in data_agg.columns:
            data_agg['rmsse'] = np.sqrt(data_agg['msse'])

        if 'total_fit_time' in data_agg.columns:
            agg_fun = get_aggregate_function(function_name)
            model_names = list(data_agg['method'].unique())
            retrain_scenarios = list(data_agg['retrain_window'].unique())
            total_fit_time = []
            for m in model_names:
                for rs in retrain_scenarios:
                    data_tmp = data \
                        .query(f'method == "{m}" and retrain_window == {rs}') \
                        .reset_index(drop = True)
                    ids_tmp = get_retrain_ids(
                        data_tmp['test_window'][0], 
                        data_tmp['horizon'][0], 
                        data_tmp['retrain_window'][0]
                    )
                    tot_fit_time_tmp = data_tmp['total_fit_time'][ids_tmp]
                    total_fit_time.append(agg_fun(tot_fit_time_tmp))
            data_agg['total_fit_time'] = total_fit_time

    return data_agg

def get_metrics(metric_names, frequency = None):

    """Function to get the evaluation metrics.

    Args:
        metric_names (list): list of evaluation metric names.
    
    Returns:
        list: list of evaluation metrics.
    """

    module_logger.info('Defining evaluation metrics...')
    freq = get_frequency(frequency)[1]

    metrics = []
    if 'bias' in metric_names:
        metrics.append(bias)
    if 'mae' in metric_names:
        metrics.append(mae)
    if 'mse' in metric_names:
        metrics.append(mse)
    if 'rmse' in metric_names:
        metrics.append(rmse)
    if 'mape' in metric_names:
        metrics.append(mape)
    if 'smape' in metric_names:
        metrics.append(smape)
    if 'mase' in metric_names:
        metrics.append(partial(mase, seasonality = freq))
    if 'msse' in metric_names:
        metrics.append(partial(msse, seasonality = freq))
    if 'rmsse' in metric_names:
        metrics.append(partial(rmsse, seasonality = freq))
    if 'ql' in metric_names:
        metrics.append(quantile_loss)
    if 'mql' in metric_names:
        metrics.append(mqloss)
    if 'sql' in metric_names:
        metrics.append(partial(scaled_quantile_loss, seasonality = freq))
    if 'smql' in metric_names:
        metrics.append(partial(scaled_mqloss, seasonality = freq))
    if 'cov' in metric_names:
        metrics.append(coverage)
    if 'cal' in metric_names:
        metrics.append(calibration)
    if 'scrps' in metric_names:
        metrics.append(scaled_crps)
    # stability metrics
    if 'stab_bias' in metric_names:
        metrics.append(bias)
    if 'mac' in metric_names:
        metrics.append(mae)
    if 'masc' in metric_names:
        metrics.append(partial(mase, seasonality = freq))
    if 'rmsc' in metric_names:
        metrics.append(rmse)
    if 'rmssc' in metric_names:
        metrics.append(partial(rmsse, seasonality = freq))
    if 'smapc' in metric_names:
        metrics.append(smape)
    if 'qc' in metric_names:
        metrics.append(quantile_loss)
    if 'mqc' in metric_names:
        metrics.append(mqloss)
    if 'sqc' in metric_names:
        metrics.append(partial(scaled_quantile_loss, seasonality = freq))
    if 'smqc' in metric_names:
        metrics.append(partial(scaled_mqloss, seasonality = freq))

    return metrics

@pf.register_dataframe_method
def evaluate_forecasts(
    out_sample_df, 
    metrics = [bias, mae, mse, rmse], 
    train_df = None,
    levels = None
):

    """Function to evaluate the point forecasts.
    
    Args:
        out_sample_df (pd.DataFrame): dataframe with columns 'unique_id', 'ds', 'y', 'fcst'.
        metrics (list): list of evaluation metrics.
        train_df (pd.DataFrame, optional): training data in the Nixtla's format. 
        Defaults to None.
        levels (list): list of levels of prediction intervals.

    Returns:
        pd.DataFrame: dataframe with evaluation results for each metric.
    """

    module_logger.info('Evaluating point forecasts...')
    eval_df = pd.DataFrame()
    # module_logger.info(f'Samlple {s} of {n_samples}...')
    eval_df = evaluate(
        out_sample_df, 
        metrics = metrics,
        models = ['fcst'],
        train_df = train_df,
        id_col = 'unique_id',
        level = levels   
    ) \
        .pivot(index = 'unique_id', columns = 'metric', values = 'fcst') \
        .reset_index()
    eval_df['method'] = out_sample_df['method'][0]
    eval_df['test_window'] = out_sample_df['test_window'][0]
    eval_df['horizon'] = out_sample_df['horizon'][0]
    eval_df['retrain_window'] = out_sample_df['retrain_window'][0]

    return eval_df

def evaluate_model(config):

    """Function to evaluate a specific model.

    Args:
        config (dict): configuration dictionary.
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
    retrain_scenarios = config['fitting']['retrain_scenarios']
    levels = config['fitting']['levels']
    combine_only = config['fitting']['combine_only']
    # model parameters
    model_names = config['model_names']
    # evaluation parameters
    eval_freq = config['evaluation']['evaluation_frequency']
    metrics = get_metrics(config['evaluation']['metrics'], eval_freq)

    # load the dataset
    if samples is not None:
        np.random.seed(seed)
    train_df = get_data(
        path_list = ['data', dataset_name],
        name_list = [dataset_name, 'prep'],
        ext = '.parquet',
        min_series_length = min_series_length, 
        max_series_length = max_series_length, 
        samples = samples
    )
    train_df = train_df[['unique_id', 'ds', 'y']]

    for m in model_names:

        module_logger.info('---------------------------- START ----------------------------')
        module_logger.info(f'[ Model name: {m} ]')

        if not combine_only:

            for rs in retrain_scenarios:

                module_logger.info(f'Evaluate predictions for retrain scenario: {rs}')
                eval_df_retrain = pd.DataFrame() # eval_df_retrain.shape[0] = 30.000 * 365 = 11.000.000
                file_names_tmp = get_file_name(
                    path_list = ['results', dataset_name, m, rs, 'outsample', 'tmp'], 
                    name_list = None,
                    ext = ext
                )
                
                for i in range(len(file_names_tmp)):
                    eval_df_tmp = load_data(
                        path_list = ['results', dataset_name, m, rs, 'outsample', 'tmp'],
                        name_list = [file_names_tmp[i]],
                        ext = ext
                    )
                    eval_df_tmp.reset_index(drop = True, inplace = True)
                    eval_df_tmp = evaluate_forecasts(
                        out_sample_df = eval_df_tmp, 
                        metrics = metrics, 
                        train_df = train_df,
                        levels = levels
                    )
                    eval_df_retrain = pd.concat([eval_df_retrain, eval_df_tmp], axis = 0)
                    del eval_df_tmp
                    if (i % 10) == 0:
                        gc.collect()

                save_data(
                    eval_df_retrain,
                    path_list = ['results', dataset_name, m, 'evaluation', 'byretrain'],
                    name_list = [dataset_name, m, rs, 'eval'],
                    ext = ext
                )
                del eval_df_retrain

        # combine and save evaluation results
        combine_and_save_files(
            path_list_to_read = ['results', dataset_name, m, 'evaluation', 'byretrain'],
            path_list_to_write = ['results', dataset_name, m, 'evaluation'],
            name_list = [dataset_name, m, 'eval'],
            ext = ext
        )
        # combine and save time results
        combine_and_save_files(
            path_list_to_read = ['results', dataset_name, m, 'time', 'byretrain'],
            path_list_to_write = ['results', dataset_name, m, 'time'],
            name_list = [dataset_name, m, 'time'],
            ext = ext
        )

        module_logger.info('----------------------------- END -----------------------------')

    module_logger.info('===============================================================')

    return

def evaluate_dataset(config):

    """Function to evaluate all models for a specific dataset.

    Args:
        config (dict): configuration dictionary.
    """
    
    module_logger.info('===============================================================')
    module_logger.info('---------------------------- START ----------------------------')

    dataset_names = config['dataset']['dataset_names']
    ext = config['dataset']['ext']
    model_names = config['model_names']

    for i in range(len(dataset_names)):

        dataset_name_tmp = dataset_names[i]
        module_logger.info(f'[ Dataset: {dataset_name_tmp} ]')

        # get file paths and names of evaluation and time samples
        eval_f_list = []
        time_f_lst = []
        for m in model_names:
            eval_f_list += [
                create_file_path(
                    path_list = ['results', dataset_name_tmp, m, 'evaluation']
                ) + 
                create_file_name(
                    name_list = [dataset_name_tmp, m, 'eval'],
                    ext = ext
                )
            ]
            time_f_lst += [
                create_file_path(
                    path_list = ['results', dataset_name_tmp, m, 'time']
                ) + 
                create_file_name(
                    name_list = [dataset_name_tmp, m, 'time'],
                    ext = ext
                )
            ]        

        # combine and save evaluation results
        combine_and_save_files(
            path_list_to_read = None,
            path_list_to_write = ['results', dataset_name_tmp, 'evaluation'],
            name_list = [dataset_name_tmp,  'eval'],
            ext = ext,  
            files_to_read = eval_f_list
        )
        # combine and save time results
        combine_and_save_files(
            path_list_to_read = None,
            path_list_to_write = ['results', dataset_name_tmp, 'evaluation'],
            name_list = [dataset_name_tmp, 'time'],
            ext = ext,
            files_to_read = time_f_lst
        )

    module_logger.info('----------------------------- END -----------------------------')
    module_logger.info('===============================================================')

    return

