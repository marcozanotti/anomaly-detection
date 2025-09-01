import sys
sys.path.insert(0, 'src/Python/utils')
import numpy as np
import pandas as pd
import pandas_flavor as pf

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
