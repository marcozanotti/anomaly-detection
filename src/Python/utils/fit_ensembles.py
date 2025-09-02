
import sys
sys.path.insert(0, 'src/Python/utils')
import numpy as np
import pandas as pd
from utilities import save_data, load_data
from evaluate_forecasts import aggregate_data

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
