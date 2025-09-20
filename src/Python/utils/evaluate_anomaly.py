import sys
sys.path.insert(0, 'src/Python/utils')
import gc
import numpy as np
import pandas as pd
import pandas_flavor as pf
from classification_losses import accuracy, recall, precision, f1, auc
from utilsforecast.evaluation import evaluate
from utilities import (
    create_file_path, create_file_name, get_file_name, 
    save_data, load_data, combine_and_save_files
)

import logging
module_logger = logging.getLogger('evaluate_anomaly')


@pf.register_dataframe_method
def aggregate_anomaly_data(
    anomaly_df, 
    group_columns, 
    drop_columns = None, 
    function_name = 'voting', 
    weights = None,
    # adjust_metrics = False
):

    """Function to aggregate evaluation metrics.

    Args:
        anomaly_df (pd.DataFrame): dataframe to aggregate.
        group_columns (list): list of columns to group by.
        function_name (str, optional): function to use to aggregate. 
        Defaults to 'mean'.
        weights (dict): dictionary of weights to use in the calculation.
        adjust_metrics (bool, optional): whether to adjust metrics. Defaults to False.
    
    Returns:
        pd.DataFrame: dataframe with aggregated data.
    """

    data_agg = anomaly_df.copy()

    module_logger.info('Aggregating data...')
    if weights is not None:
        logging.info('Normalizing weights...')
        weights = {k: v / sum(weights.values()) for k, v in weights.items()}
        weights_df = pd.DataFrame(list(weights.items()), columns = ['method', 'weight'])
        data_agg = data_agg.merge(weights_df, on = 'method', how = 'left')

    if drop_columns is not None:
        data_agg.drop(columns = drop_columns, inplace = True)

    if function_name == 'voting':

        data_agg = data_agg \
            .groupby(group_columns) \
            .agg('mean') \
            .reset_index() \
            .rename(columns = {'anomaly': 'score'})

    elif function_name == 'wvoting':

        data_agg = data_agg \
            .groupby(group_columns) \
            .apply(lambda x: np.average(x.anomaly, weights = x.weight), include_groups = False) \
            .reset_index() \
            .rename(columns = {0: 'score'})
        
    else:
        logging.error(f'Unknown function_name {function_name}.')
        raise ValueError(f'Invalid function_name: {function_name}')

    return data_agg

def get_anomaly_metrics(metric_names):

    """Function to get the evaluation metrics.

    Args:
        metric_names (list): list of evaluation metric names.
    
    Returns:
        list: list of evaluation metrics.
    """

    module_logger.info('Defining evaluation metrics...')

    metrics = []
    if 'accuracy' in metric_names:
        metrics.append(accuracy)
    if 'recall' in metric_names:
        metrics.append(recall)
    if 'precision' in metric_names:
        metrics.append(precision)
    if 'f1' in metric_names:
        metrics.append(f1)
    if 'auc' in metric_names:
        metrics.append(auc)

    return metrics

@pf.register_dataframe_method
def evaluate_anomaly(
    out_sample_df, 
    metrics = [accuracy, recall, precision, f1],
    levels = None
):

    """Function to evaluate the anomalies.
    
    Args:
        out_sample_df (pd.DataFrame): dataframe with columns 'unique_id', 'ds', 'is_real_anomaly', 'anomaly'.
        metrics (list): list of evaluation metrics.
        levels (list): list of levels of prediction intervals.

    Returns:
        pd.DataFrame: dataframe with evaluation results for each metric.
    """

    module_logger.info('Evaluating anomalies...')

    if levels is not None:

        interval_cols = [c for c in out_sample_df.columns if '-intervals-' in c]
        score_cols = [c for c in out_sample_df.columns if '-score-' in c]
        models = interval_cols

        eval_df = evaluate(
            out_sample_df, 
            metrics = metrics,
            models = models,
            id_col = 'unique_id',
            target_col = 'is_real_anomaly'
        )

        if score_cols:
            eval_score_df = evaluate(
                out_sample_df, 
                metrics = metrics,
                models = score_cols,
                id_col = 'unique_id',
                target_col = 'is_real_anomaly'
            )
            eval_df = eval_df.merge(eval_score_df, how = 'left', on = ['unique_id', 'metric'])
            models = interval_cols + score_cols

        eval_df = eval_df.melt(
            id_vars = ['unique_id', 'metric'],
            value_vars = models,
            var_name = 'type',
            value_name = 'anomaly'
        )
        eval_df['type'] = eval_df['type'].str.replace('anomaly-', '')
        eval_df['level'] = eval_df['type'].str.replace(r'\w+-', '', regex = True).astype(float)
        eval_df['type'] = eval_df['type'].str.replace(r'-\d+\.?\d+?', '', regex = True)
        eval_df = eval_df.pivot(index = ['unique_id', 'type', 'level'], columns = 'metric', values = 'anomaly')

    else:

        models = ['anomaly']
        eval_df = evaluate(
            out_sample_df, 
            metrics = metrics,
            models = models,
            id_col = 'unique_id',
            target_col = 'is_real_anomaly'
        )
        eval_df['type'] = 'score'
        eval_df = eval_df.pivot(index = ['unique_id', 'type'], columns = 'metric', values = 'anomaly')

    eval_df.reset_index(inplace = True)
    eval_df['method'] = out_sample_df['method'][0]
    eval_df['test_window'] = out_sample_df['test_window'][0]
    eval_df['horizon'] = out_sample_df['horizon'][0]
    eval_df['retrain_window'] = out_sample_df['retrain_window'][0]

    return eval_df

def evaluate_anomaly_model(config):

    """Function to evaluate anomalies of a specific model.

    Args:
        config (dict): configuration dictionary.
    """

    module_logger.info('===============================================================')

    # dataset parameters
    dataset_name = config['dataset']['dataset_name']
    ext = config['dataset']['ext']
    # fitting parameters    
    retrain_scenarios = config['fitting']['retrain_scenarios']
    levels = config['fitting']['levels']
    combine_only = config['fitting']['combine_only']
    # model parameters
    model_names = config['model_names']
    # evaluation parameters
    metrics = get_anomaly_metrics(config['evaluation']['metrics'])

    for m in model_names:

        module_logger.info('---------------------------- START ----------------------------')
        module_logger.info(f'[ Model name: {m} ]')

        if not combine_only:

            for rs in retrain_scenarios:

                module_logger.info(f'Evaluate anomalies for retrain scenario: {rs}')
                eval_df_retrain = pd.DataFrame() 
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
                    eval_df_tmp = evaluate_anomaly(
                        out_sample_df = eval_df_tmp, 
                        metrics = metrics,
                        levels = levels
                    )
                    eval_df_retrain = pd.concat([eval_df_retrain, eval_df_tmp], axis = 0)
                    del eval_df_tmp
                    if (i % 10) == 0:
                        gc.collect()

                save_data(
                    eval_df_retrain,
                    path_list = ['results', dataset_name, m, 'evaluation', 'byretrain'],
                    name_list = [dataset_name, m, rs, 'eval', 'anomaly'],
                    ext = ext
                )
                del eval_df_retrain

        # combine and save evaluation results
        combine_and_save_files(
            path_list_to_read = ['results', dataset_name, m, 'evaluation', 'byretrain'],
            path_list_to_write = ['results', dataset_name, m, 'evaluation'],
            name_list = [dataset_name, m, 'eval', 'anomaly'],
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

def evaluate_anomaly_dataset(config):

    """Function to evaluate all anomaly models for a specific dataset.

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
                    name_list = [dataset_name_tmp, m, 'eval', 'anomaly'],
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
            name_list = [dataset_name_tmp, 'eval', 'anomaly'],
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