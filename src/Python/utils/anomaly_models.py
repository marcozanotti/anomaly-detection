import numpy as np
import pandas as pd
import pandas_flavor as pf

import logging
module_logger = logging.getLogger('anomaly_models')


@pf.register_dataframe_method
def Zscore(df, q = 3):

    """Function to detect anomalies using Z-Score method.

    Args:
        df (pd.DataFrame): dataframe in Nixtla's format.
        q (int): the quantile level. Default to 3.
    
    Returns:
        pd.DataFrame: dataframe with anomaly data.
    """

    data_anom = df.copy()

    module_logger.info('Computing anomalies with Z-Score method...')

    z = (data_anom['y'] - np.mean(data_anom['y'])) / np.std(data_anom['y'])
    is_anomaly = np.abs(z) > q
    data_anom['Zscore'] = is_anomaly.astype('int')

    return data_anom
