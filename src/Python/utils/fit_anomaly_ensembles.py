
import sys
sys.path.insert(0, 'src/Python/utils')
import numpy as np
import pandas as pd
import pandas_flavor as pf
from utilities import save_data, load_data
from evaluate_anomaly import aggregate_anomaly_data
from evaluate_forecasts import aggregate_data

import logging
module_logger = logging.getLogger('fit_anomaly_ensembles')


@pf.register_dataframe_method
def anomaly_select(anomaly_df, threshold = 0.5):

    """
    Select anomalies based on anomaly score.

    Args:
        anomaly_df (pd.DataFrame): DataFrame with score column.
        threshold (float): threshold for selecting anomalies (default = 0.5).

    Returns:
        pd.DataFrame: Original DataFrame with an additional 'anomaly' column.
    """

    logging.info('Selecting anomalies based on anomaly score...')
    logging.info(f'Threshold = {threshold}')
    score_df = anomaly_df.copy()
    score_df['anomaly'] = (anomaly_df['score'] >= threshold).astype(int)

    return score_df

def fit_anomaly_ensembles(config):

    """Function to create ensembles of anomaly models.

    Args:
        config (dict): Configuration dictionary.
    """

    module_logger.info('===============================================================')
    module_logger.info('---------------------------- START ----------------------------')

    # dataset parameters
    dataset_name = config['dataset']['dataset_name']
    # frequency = config['dataset']['frequency']
    ext = config['dataset']['ext']
    # fitting parameters
    # test_window = config['fitting']['test_window']
    # horizon = config['fitting']['horizon']
    retrain_scenarios = config['fitting']['retrain_scenarios']
    # model parameters
    ensemble_models_list = config['ensembling']['model_names']
    ensemble_names_list = config['ensembling']['name']
    ensemble_methods = config['ensembling']['methods']
    ensemble_models_weights_list = config['ensembling']['model_weights']
    ensemble_anomaly_threshold = config['ensembling']['anomaly_threshold']

    for i in range(len(ensemble_models_list)):

        model_names = ensemble_models_list[i]
        ensemble_name = ensemble_names_list[i]
        module_logger.info(f'[ Ensemble: {ensemble_name} | Models: {model_names} ]')

        # compute and save the ensemble predictions
        for rs in retrain_scenarios:

            module_logger.info(f'[ Retrain scenario: {rs} ]')
            outsample_df_tmp = pd.DataFrame()
            time_df_tmp = pd.DataFrame()

            for m in model_names:

                outsample_df_model_tmp = load_data(
                    path_list = ['results', dataset_name, m, rs, 'outsample', 'tmp'],
                    name_list = [dataset_name, m, rs, 'outsample'],
                    ext = ext
                )
                outsample_df_tmp = pd.concat([outsample_df_tmp, outsample_df_model_tmp], axis = 0)
                outsample_df_tmp.reset_index(drop = True, inplace = True)
                del outsample_df_model_tmp

                time_df_model_tmp = load_data(
                    path_list = ['results', dataset_name, m, 'time', 'byretrain'],
                    name_list = [dataset_name, m, rs, 'time'],
                    ext = ext
                )
                time_df_tmp = pd.concat([time_df_tmp, time_df_model_tmp], axis = 0)
                del time_df_model_tmp

            for ens in ensemble_methods:

                module_logger.info(f'Computing ensemble {ens} anomalies...')
                ensemble_name_tmp = 'Ensemble' + ens.capitalize() + ensemble_name 
                if ens == 'voting':
                    w = None
                else:
                    w = ensemble_models_weights_list[ensemble_name]
                
                ensemble_df_tmp = aggregate_anomaly_data(
                    anomaly_df = outsample_df_tmp,
                    group_columns = ['sample', 'test_window', 'horizon', 'retrain_window', 'ds', 'unique_id'],
                    drop_columns = ['method', 'y', 'is_real_anomaly'], 
                    function_name = ens,
                    weights = w
                )
                ensemble_df_tmp = anomaly_select(ensemble_df_tmp, threshold = ensemble_anomaly_threshold)
                ensemble_df_tmp['method'] = ensemble_name_tmp
                ensemble_df_tmp = ensemble_df_tmp.merge(
                    outsample_df_tmp[['unique_id', 'ds', 'y', 'is_real_anomaly']].drop_duplicates(), 
                    how = 'left', 
                    on = ['unique_id', 'ds'], 
                    copy = False
                )
                ensemble_df_tmp = ensemble_df_tmp[list(outsample_df_tmp.columns) + ['score']]
                ensemble_df_tmp.sort_values(by = ['unique_id', 'ds'], inplace = True)
                save_data(
                    data = ensemble_df_tmp, 
                    path_list = ['results', dataset_name, ensemble_name_tmp, rs, 'outsample', 'tmp'],
                    name_list = [dataset_name, ensemble_name_tmp, rs, 'outsample'],
                    ext = ext
                )
                del ensemble_df_tmp

                module_logger.info(f'Computing time of ensemble {ens}...')
                ensemble_time_df_tmp = aggregate_data(
                    data = time_df_tmp,
                    group_columns = ['sample', 'test_window', 'horizon', 'retrain_window', 'unique_id'],
                    drop_columns = ['method'], 
                    function_name = 'sum',
                    adjust_metrics = False
                )
                ensemble_time_df_tmp['method'] = ensemble_name_tmp
                ensemble_time_df_tmp = ensemble_time_df_tmp[list(time_df_tmp.columns)]
                ensemble_time_df_tmp.sort_values(by = ['unique_id', 'sample'], inplace = True)
                save_data(
                    data = ensemble_time_df_tmp, 
                    path_list = ['results', dataset_name, ensemble_name_tmp, 'time', 'byretrain'],
                    name_list = [dataset_name, ensemble_name_tmp, rs, 'time'],
                    ext = ext
                )
                del ensemble_time_df_tmp
            
            del outsample_df_tmp, time_df_tmp

    module_logger.info('----------------------------- END -----------------------------')
    module_logger.info('===============================================================')

    return
