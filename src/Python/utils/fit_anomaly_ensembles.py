
import sys
sys.path.insert(0, 'src/Python/utils')
import numpy as np
import pandas as pd
from utilities import save_data, load_data
from evaluate_forecasts import aggregate_data

import logging
module_logger = logging.getLogger('fit_anomaly_ensembles')


def anomaly_score(anomaly_df, weights = None):

    """
    Compute anomaly score (unweighted or weighted).

    Args:
        anomaly_data (pd.DataFrame): DataFrame with anomaly indicator columns (0/1).
        weights (list or np.ndarray, optional): Weights for each column. Will be normalized.

    Returns:
        pd.Series: anomaly score per row.
    """

    if weights is None:
        logging.info('Computing anomaly score...')
        score = anomaly_df.sum(axis=1) / anomaly_df.shape[1]
    else:
        logging.info('Computing anomaly weighted score...')
        logging.info(f'Original Weights = {weights}')
        weights = np.array(weights, dtype = float)
        weights = weights / weights.sum()
        logging.info(f'Normalized Weights = {weights}')
        weighted_data = anomaly_df.mul(weights, axis=1)
        score = weighted_data.sum(axis=1)

    return score

def anomaly_select(score, threshold = 0.5):

    """
    Select anomalies based on anomaly score.

    Args:
        score (array-like or pd.Series): anomaly scores.
        threshold (float): threshold for selecting anomalies (default = 0.5).

    Returns:
        np.ndarray: binary array (1 = anomaly, 0 = normal).
    """

    logging.info('Selecting anomalies based on anomaly score...')
    logging.info(f'Threshold = {threshold}')

    is_anomaly = (score > threshold).astype(int)
    return is_anomaly

def ensemble_anomalies(df, methods, ensemble_type = 'voting', threshold = 0.5, weights = None):

    """
    Estimate ensemble anomalies using voting or weighted voting.

    Args:
        df (pd.DataFrame): Input DataFrame with anomaly indicator columns.
        methods (list): List of column names to include in the ensemble.
        ensemble_type (str, optional): 'voting' or 'weighted_voting'. Defaults to 'voting'.
        threshold (float, optional): Threshold for anomaly selection. Defaults to 0.5.
        weights (list or np.ndarray, optional): Weights for weighted voting.

    Returns:
        pd.DataFrame: Original DataFrame with an additional 'ensemble' column.
    """

    logging.info('Estimating ensemble anomalies...')
    logging.info(f' [ Ensemble Method: {ensemble_type} ] ')

    # wide = df.pivot(index=id_col, columns=method_col, values=value_col)

    anomaly_data = df[methods].copy()

    if ensemble_type == 'voting':
        score = anomaly_score(anomaly_data)
    elif ensemble_type == 'weighted_voting':
        score = anomaly_score(anomaly_data, weights=weights)
    else:
        logging.error(f'Unknown ensemble_type {ensemble_type}.')
        raise ValueError(f'Invalid ensemble_type: {ensemble_type}')

    is_anomaly = anomaly_select(score, threshold = threshold)

    res_df = df.copy()
    res_df['ensemble'] = is_anomaly

    return res_df

def fit_anomaly_ensembles(config):

    """Function to create ensembles of models.

    Args:
        config (dict): Configuration dictionary.
    """

    module_logger.info('===============================================================')
    module_logger.info('---------------------------- START ----------------------------')

    # dataset parameters
    dataset_name = config['dataset']['dataset_name']
    frequency = config['dataset']['frequency']
    ext = config['dataset']['ext']
    # fitting parameters
    test_window = config['fitting']['test_window']
    horizon = config['fitting']['horizon']
    retrain_scenarios = config['fitting']['retrain_scenarios']
    # model parameters
    ensemble_models_list = config['ensembling']['model_names']
    ensemble_methods = config['ensembling']['methods']
    ensemble_names_list = config['ensembling']['name']

    n_samples = test_window - horizon + 1


    for i in range(len(ensemble_models_list)):

        model_names = ensemble_models_list[i]
        ensemble_name = ensemble_names_list[i]
        module_logger.info(f'[ Ensemble: {ensemble_name} ]')

        # compute and save the ensemble predictions
        for rs in retrain_scenarios:

            module_logger.info(f'[ Retrain scenario: {rs} ]')

            for s in range(n_samples):

                outsample_df_sample_tmp = pd.DataFrame()

                for m in model_names:

                    outsample_df_model_tmp = load_data(
                        path_list = ['results', dataset_name, frequency, m, rs, 'outsample', 'tmp'],
                        name_list = [dataset_name, frequency, m, rs, 'outsample', s],
                        ext = ext
                    )
                    outsample_df_sample_tmp = pd.concat([outsample_df_sample_tmp, outsample_df_model_tmp], axis = 0)
                    del outsample_df_model_tmp

                for ens in ensemble_methods:

                    module_logger.info(f'Computing ensemble {ens} predictions...')

                    ensemble_name_tmp = 'Ensemble' + ens.capitalize() + ensemble_name 

                    ensemble_df_tmp = aggregate_data(
                        data = outsample_df_sample_tmp,
                        group_columns = ['sample', 'test_window', 'horizon', 'retrain_window', 'ds', 'unique_id'],
                        drop_columns = ['method', 'y'], 
                        function_name = ens,
                        adjust_metrics = False
                    )
                    ensemble_df_tmp['method'] = ensemble_name_tmp
                    ensemble_df_tmp = ensemble_df_tmp.merge(
                        outsample_df_sample_tmp[['unique_id', 'ds', 'y']].drop_duplicates(), 
                        how = 'left', 
                        on = ['unique_id', 'ds'], 
                        copy = False
                    )

                    save_data(
                        data = ensemble_df_tmp, 
                        path_list = ['results', dataset_name, frequency, ensemble_name_tmp, rs, 'outsample', 'tmp'],
                        name_list = [dataset_name, frequency, ensemble_name_tmp, rs, 'outsample', s],
                        ext = ext
                    )
                    del ensemble_df_tmp
                
                del outsample_df_sample_tmp

        # compute and save the ensemble time results
        module_logger.info('---------------------------------------------------------------')
        for rs in retrain_scenarios:

            for ens in ensemble_methods:

                module_logger.info(f'Computing ensemble {ens} time results...')
                time_df_tmp = pd.DataFrame()

                for m in model_names:
                    time_df_model_tmp = load_data(
                        path_list = ['results', dataset_name, frequency, m, 'time', 'byretrain'],
                        name_list = [dataset_name, frequency, m, rs, 'time'],
                        ext = ext
                    )
                    time_df_tmp = pd.concat([time_df_tmp, time_df_model_tmp], axis = 0)
                    del time_df_model_tmp
                
                ensemble_name_tmp = 'Ensemble' + ens.capitalize() + ensemble_name
                ensemble_time_df_tmp = aggregate_data(
                    data = time_df_tmp,
                    group_columns = ['sample', 'test_window', 'horizon', 'retrain_window'],
                    drop_columns = ['method'], 
                    function_name = 'sum',
                    adjust_metrics = False
                )
                ensemble_time_df_tmp['method'] = ensemble_name_tmp
                save_data(
                    data = ensemble_time_df_tmp, 
                    path_list = ['results', dataset_name, frequency, ensemble_name_tmp, 'time', 'byretrain'],
                    name_list = [dataset_name, frequency, ensemble_name_tmp, rs, 'time'],
                    ext = ext
                )
                del time_df_tmp, ensemble_time_df_tmp

    module_logger.info('----------------------------- END -----------------------------')
    module_logger.info('===============================================================')

    return
