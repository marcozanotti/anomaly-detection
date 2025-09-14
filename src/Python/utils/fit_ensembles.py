
import sys
sys.path.insert(0, 'src/Python/utils')
import numpy as np
import pandas as pd
from utilities import save_data, load_data
from evaluate_forecasts import aggregate_data
from fit_models import add_anomaly

import logging
module_logger = logging.getLogger('fit_ensembles')


def fit_ensembles(config):

    """Function to create ensembles of models.

    Args:
        config (dict): Configuration dictionary.
    """

    module_logger.info('===============================================================')
    module_logger.info('---------------------------- START ----------------------------')

    # dataset parameters
    dataset_name = config['dataset']['dataset_name']
    ext = config['dataset']['ext']
    # fitting parameters
    retrain_scenarios = config['fitting']['retrain_scenarios']
    levels = config['fitting']['levels']
    # model parameters
    ensemble_models_list = config['ensembling']['model_names']
    ensemble_methods = config['ensembling']['methods']
    ensemble_names_list = config['ensembling']['name']
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
                    path_list = ['results', dataset_name,m, rs, 'outsample', 'tmp'],
                    name_list = [dataset_name, m, rs, 'outsample'],
                    ext = ext
                )
                outsample_df_tmp = pd.concat([outsample_df_tmp, outsample_df_model_tmp], axis = 0)
                del outsample_df_model_tmp

                time_df_model_tmp = load_data(
                    path_list = ['results', dataset_name, m, 'time', 'byretrain'],
                    name_list = [dataset_name, m, rs, 'time'],
                    ext = ext
                )
                time_df_tmp = pd.concat([time_df_tmp, time_df_model_tmp], axis = 0)
                del time_df_model_tmp
            
            outsample_df_tmp.reset_index(drop = True, inplace = True)
            time_df_tmp.reset_index(drop = True, inplace = True)

            for ens in ensemble_methods:

                module_logger.info(f'Computing ensemble {ens} predictions...')
                ensemble_name_tmp = 'Ensemble' + ens.capitalize() + ensemble_name 

                ensemble_df_tmp = aggregate_data(
                    data = outsample_df_tmp,
                    group_columns = ['sample', 'test_window', 'horizon', 'retrain_window', 'ds', 'unique_id'],
                    drop_columns = ['method', 'y', 'is_real_anomaly'], 
                    function_name = ens,
                    adjust_metrics = False
                )

                ensemble_df_tmp['method'] = ensemble_name_tmp
                ensemble_df_tmp = ensemble_df_tmp.merge(
                    outsample_df_tmp[['unique_id', 'ds', 'y', 'is_real_anomaly']].drop_duplicates(), 
                    how = 'left', 
                    on = ['unique_id', 'ds'], 
                    copy = False
                )

                ensemble_df_tmp.rename(columns = {f'anomaly-intervals-{lvl}': f'score-{lvl}' for lvl in levels}, inplace = True)
                ensemble_df_tmp = add_anomaly(ensemble_df_tmp, type = 'intervals', levels = levels)
                ensemble_df_tmp = add_anomaly(ensemble_df_tmp, type = 'score', levels = levels, threshold = ensemble_anomaly_threshold)
                ensemble_df_tmp.sort_values(by = ['unique_id', 'ds'], inplace = True)
                ensemble_df_tmp.reset_index(drop = True, inplace = True)
                save_data(
                    data = ensemble_df_tmp, 
                    path_list = ['results', dataset_name, ensemble_name_tmp, rs, 'outsample', 'tmp'],
                    name_list = [dataset_name, ensemble_name_tmp, rs, 'outsample'],
                    ext = ext
                )
                del ensemble_df_tmp

                module_logger.info(f'Computing ensemble {ens} time results...')
                ensemble_time_df_tmp = aggregate_data(
                    data = time_df_tmp,
                    group_columns = ['sample', 'test_window', 'horizon', 'retrain_window', 'unique_id'],
                    drop_columns = ['method'], 
                    function_name = 'sum',
                    adjust_metrics = False
                )
                ensemble_time_df_tmp['method'] = ensemble_name_tmp
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
