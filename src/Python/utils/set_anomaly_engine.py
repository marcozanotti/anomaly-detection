
import sys
sys.path.insert(0, 'src/Python/utils')
from functools import partial
from anomaly_models import (
    Zscore, Pierce, Chauvenet, Grubbs, Dixon, Tukey, Barbato, Hampel
)

import logging
module_logger = logging.getLogger('set_anomaly_engine')

def get_anomaly_model_type(model_name):

    """Function to get the model type.

    Args:
        model_name (str): name of the model.
    
    Returns:
        str: model type.
    """
    
    # statistical anomaly detection models
    sa = ['Zscore', 'Pierce', 'Chauvenet', 'Grubbs', 'Dixon']
    # box-plot anomaly detection models
    ba = ['Tukey', 'Barbato']
    # time-series anomaly detection models
    tsa = ['Hampel']

    if model_name in sa:
        model_type = 'Statistical'
    elif model_name in ba:
        model_type = 'Box-plot'
    elif model_name in tsa:
        model_type = 'Time-series'
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return model_type

def get_default_anomaly_model_params(model_name):
    
    """Function to get the default parameters for the models.

    Args:
        model_name (str): name of the model.
    
    Returns:
        dict: default model parameters.
    """

    module_logger.info('Defining default model parameters...')

    if model_name == 'Zscore':
        model_params = {
            model_name: {
                'q': 3
            }
        }
    elif model_name == 'Pierce':
        model_params = {
            model_name: {None}
        }
    elif model_name == 'Chauvenet':
        model_params = {
            model_name: {None}
        }
    elif model_name == 'Grubbs':
        model_params = {
            model_name: {
                'alpha': 0.05
            }
        }
    elif model_name == 'Dixon':
        model_params = {
            model_name: {
                'alpha': 0.05,
                'two_sided': True
            }
        }
    elif model_name == 'Tukey':
        model_params = {
            model_name: {
                'extreme': False
            }
        }
    elif model_name == 'Barbato':
        model_params = {
            model_name: {
                'extreme': False
            }
        }
    elif model_name == 'Hampel':
        model_params = {
            model_name: {
                'bandwidth': 3,
                'k': 3
            }
        }
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return model_params

def set_anomaly_model(model_name, model_params = None):

    """Function to get the models.

    Args:
        model_name (str): name of the model.
        model_params (dict, optional): parameters for the model. Defaults to None.
    
    Returns:
        list: model.
    """

    module_logger.info('Defining the model...')
    
    if model_params is None:
        model_params = get_default_anomaly_model_params(model_name)[model_name]

    module_logger.info(f'Model parameters: {model_params}')

    if model_name == 'Zscore':
        model = partial(Zscore, **model_params)
    elif model_name == 'Pierce':
        model = partial(Pierce)
    elif model_name == 'Chauvenet':
        model = partial(Chauvenet)
    elif model_name == 'Grubbs':
        model = partial(Grubbs, **model_params)
    elif model_name == 'Dixon':
        model = partial(Dixon, **model_params)
    elif model_name == 'Tukey':
        model = partial(Tukey, **model_params)
    elif model_name == 'Barbato':
        model = partial(Barbato, **model_params)
    elif model_name == 'Hampel':
        model = partial(Hampel, **model_params)
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return model

def set_anomaly_engine(model_name, model_params = None):

    """Function to set the engine based on the model name.

    Args:
        model_name (str): name of the model.
        model_params (dict, optional): parameters for the model. Defaults to None.
    
    Returns:
        StatisticalAnomaly: StatsAnomaly engine.
    """

    module_logger.info('Setting the engine...')
    model = set_anomaly_model(model_name, model_params)
    engine = model

    return engine
