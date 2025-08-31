
import sys
sys.path.insert(0, 'src/Python/utils')
from functools import partial
from anomaly_models import Zscore

import logging
module_logger = logging.getLogger('set_anomaly_engine')

def get_anomaly_model_type(model_name):

    """Function to get the model type.

    Args:
        model_name (str): name of the model.
    
    Returns:
        str: model type.
    """
    
    sa = ['Zscore']

    if model_name in sa:
        model_type = 'sa'
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
    
    model_type = get_anomaly_model_type(model_name)
    
    if model_params is None:
        model_params = get_default_anomaly_model_params(model_name)[model_name]

    module_logger.info(f'Model parameters: {model_params}')

    if model_type == 'sa':

        if model_name == 'Zscore':
            model = partial(Zscore, **model_params)
        else:
            raise ValueError(f'Invalid model: {model_name}')
    
    else:
        raise ValueError(f'Invalid model type: {model_type}')

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
    model_type = get_anomaly_model_type(model_name)
    model = set_anomaly_model(model_name, model_params)

    if model_type == 'sa':

        engine = model
        
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return engine
