
import sys
sys.path.insert(0, 'src/Python/utils')
import os
import pandas_flavor as pf
from pandas.api.types import is_numeric_dtype
from mlforecast import MLForecast
from neuralforecast import NeuralForecast
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from neuralforecast.models import MLP, LSTM, TCN, NBEATSx, NHITS

# NOTE: feature and transform functions must be imported to be used with eval('fun_name')
# from sklearn.preprocessing import FunctionTransformer
# from mlforecast.target_transforms import GlobalSklearnTransformer
# from mlforecast.target_transforms import LocalStandardScaler, LocalMinMaxScaler, Differences
from mlforecast.lag_transforms import RollingMean, ExpandingMean
from neuralforecast.losses.pytorch import MAE, MSE, RMSE
from custom_feats import is_weekend
from utilities import get_frequency

import logging
module_logger = logging.getLogger('set_engine')

def get_loss_function(loss):
    """Function to get the loss function.

    Args:
        loss (str): name of the loss function.
    
    Returns:
        function: loss function.
    """

    module_logger.info('Defining loss function...')

    if loss is not None:
        try:
            fun_tmp = eval(loss)
        except:
            fun_tmp = 'error'
        if fun_tmp != 'error':
            loss = fun_tmp

    return loss

def get_target_transforms(target_transforms):
    """Function to get the target transforms for the dataset.

    Args:
        target_transforms (list): list of target transformations.
    
    Returns:
        list: target transforms.
    """

    module_logger.info('Defining target trasformations...')
    # Log1p = FunctionTransformer(func = np.log1p, inverse_func = np.expm1)
    # target_transforms = [GlobalSklearnTransformer(Log1p)],
    
    if target_transforms is not None:
        for i in range(len(target_transforms)):
            try:
                fun_tmp = eval(target_transforms[i])
            except:
                fun_tmp = 'error'
            if fun_tmp != 'error':
                target_transforms[i] = fun_tmp

    return target_transforms

def get_lags(feature_list):
    """Function to get the lags for the dataset.

    Args:
        feature_list (list): list of features.
    
    Returns:
        list: lags.
    """

    module_logger.info('Defining lags...')
    return feature_list

def get_lag_transforms(feature_list):
    """Function to get the lag transforms for the dataset.

    Args:
        feature_list (list): list of features.
    
    Returns:
        dict: lag transforms.
    """

    module_logger.info('Defining lag trasformations...')
    if feature_list is not None:
        for k in feature_list.keys():
            for i in range(len(feature_list[k])):
                try:
                    fun_tmp = eval(feature_list[k][i])
                except:
                    fun_tmp = 'error'
                if fun_tmp != 'error':
                    feature_list[k][i] = fun_tmp

    return feature_list

def get_date_features(feature_list):
    """Function to get the date features for the dataset.

    Args:
        feature_list (list): list of features.
    
    Returns:
        list: date features.
    """

    module_logger.info('Defining date features...')

    if feature_list is not None:
        for i in range(len(feature_list)):
            try:
                fun_tmp = eval(feature_list[i])
            except:
                fun_tmp = 'error'
            if fun_tmp != 'error':
                feature_list[i] = fun_tmp

    return feature_list

def get_model_type(model_name):
    """Function to get the model type for the dataset.

    Args:
        model_name (str): name of the model.
    
    Returns:
        str: model type.
    """
    
    sf = ['ETS', 'ARIMA']
    ml = [
        'LinearRegression', 'Lasso', 'Ridge', 
        'RandomForestRegressor', 
        'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor' 
    ]
    dl = ['MLP', 'LSTM', 'TCN', 'NBEATSx', 'NHITS']

    if model_name in sf:
        model_type = 'sf'
    elif model_name in ml:
        model_type ='ml'
    elif model_name in dl:
        model_type = 'dl'
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return model_type

def get_default_model_params(model_name):
    """Function to get the default parameters for the models.

    Args:
        model_name (str): name of the model.
    
    Returns:
        dict: default model parameters.
    """

    module_logger.info('Defining default model parameters...')

    if model_name == 'LinearRegression':

        model_params = {
            model_name: {'n_jobs': -1}
        }
    
    elif model_name == 'Lasso':

        model_params = {
            model_name: {}
        }
    
    elif model_name == 'Ridge':

        model_params = {
            model_name: {}
        }
    
    elif model_name == 'RandomForestRegressor':

        model_params = {
            model_name: {
                'n_estimators': 100,
                'n_jobs': -1
            }
        }
    
    elif model_name == 'XGBRegressor':

        model_params = {
            model_name: {
                'n_estimators': 100,
                'n_jobs': -1
            }
        }
    
    elif model_name == 'LGBMRegressor':

        model_params = {
            model_name: {
                'n_estimators': 100,
                'n_jobs': -1
            }
        }

    elif model_name == 'CatBoostRegressor':

        model_params = {
            model_name: {
                'n_estimators': 100,
                'thread_count': -1
            }
        }
    
    elif model_name == 'MLP':

        model_params = {
            model_name: {
                'h': 28,
                'input_size': 2,
                'max_steps': 100,
                'early_stop_patience_steps': 10
            }
        }

    elif model_name == 'LSTM':

        model_params = {
            model_name: {
                'h': 28,
                'max_steps': 100,
                'early_stop_patience_steps': 10
            }
        }
    
    elif model_name == 'TCN':

        model_params = {
            model_name: {
                'h': 28,
                'max_steps': 100,
                'early_stop_patience_steps': 10
            }
        }
    
    elif model_name == 'NBEATSx':

        model_params = {
            model_name: {
                'h': 28,
                'input_size': 7,
                'max_steps': 100,
                'early_stop_patience_steps': 10
            }
        }

    elif model_name == 'NHITS':

        model_params = {
            model_name: {
                'h': 28,
                'input_size': 7,
                'max_steps': 100,
                'early_stop_patience_steps': 10 
            }
        }
    
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return model_params

def set_model(model_name, model_params = None):
    """Function to get the models for the dataset.

    Args:
        model_name (str): name of the model.
        model_params (dict, optional): parameters for the model. Defaults to None.
    
    Returns:
        list: model.
    """

    module_logger.info('Defining the model...')
    model_type = get_model_type(model_name)
    if model_params is None:
        model_params = get_default_model_params(model_name)[model_name]
    module_logger.info(f'Model parameters: {model_params}')

    if model_type == 'ml':

        if model_name == 'LinearRegression':
            model = [LinearRegression(**model_params)]
        elif model_name == 'Lasso':
            model = [Lasso(**model_params)]
        elif model_name == 'Ridge':
            model = [Ridge(**model_params)]
        elif model_name == 'RandomForestRegressor':
            model = [RandomForestRegressor(**model_params)]
        elif model_name == 'XGBRegressor':
            model = [XGBRegressor(**model_params)]
        elif model_name == 'LGBMRegressor':
            model = [LGBMRegressor(**model_params)]
        elif model_name == 'CatBoostRegressor':
            model = [CatBoostRegressor(**model_params)]
        else:
            raise ValueError(f'Invalid model: {model_name}')

    elif model_type == 'dl':

        if 'loss' in model_params.keys():
            model_params['loss'] = get_loss_function(model_params['loss'])

        if model_name == 'MLP':
            model = [MLP(**model_params)]
        elif model_name == 'LSTM':
            model = [LSTM(**model_params)]
        elif model_name == 'TCN':
            model = [TCN(**model_params)]
        elif model_name == 'NBEATSx':
            model = [NBEATSx(**model_params)]
        elif model_name == 'NHITS':
            model = [NHITS(**model_params)]
        else:
            raise ValueError(f'Invalid model: {model_name}')
    
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    return model

def set_engine(model_name, frequency, features, target_transforms = None, model_params = None):

    """Function to set the engine based on the model name.

    Args:
        model_name (str): name of the model.
        frequency (str): frequency of the dataset.
        features (dict): features of the dataset.
        target_transforms (list, optional): target transformations. Defaults to None.
        model_params (dict, optional): parameters for the model. Defaults to None.
    
    Returns:
        StatsForecast: StatsForecast engine.
        MLForecast: MLForecast engine.
    """

    module_logger.info('Setting the engine...')
    model_type = get_model_type(model_name)
    model = set_model(model_name, model_params)

    freq = get_frequency(frequency)[0]
    target_transforms = get_target_transforms(target_transforms = target_transforms)
    lags = get_lags(feature_list = features['lags'])
    lag_transforms = get_lag_transforms(feature_list = features['lag_transforms'])
    date_features = get_date_features(feature_list = features['date'])

    if model_type =='sf':

        raise ValueError(f'Not yet implemented for model {model_name}')

    elif model_type == 'ml':

        engine = MLForecast(
            models = model,
            freq = freq, 
            num_threads = os.cpu_count(),
            target_transforms = target_transforms,
            lags = lags,
            lag_transforms = lag_transforms,
            date_features = date_features
        )
    
    elif model_type == 'dl':

        engine = NeuralForecast(
            models = model, 
            freq = freq,
            local_scaler_type = target_transforms
        )
        
    else:
        raise ValueError(f'Invalid model: {model_name}')

    return engine

@pf.register_dataframe_method
def add_data_features(data, frequency, features, remove_static = False):

    """Function to add features to the data.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        frequency (string): The frequency of the data (e.g., 'daily', 'weekly').
        features (dict): Dictionary containing feature details.
        forced_frequency (num, optional): value of the frequency to force. Defaults to None. 
    
    Returns:
        pd.DataFrame: dataframe with date features added.
    """

    module_logger.info(f'Adding features to the dataset...')

    freq = get_frequency(frequency)[0]
    lags = get_lags(feature_list = features['lags'])
    lag_transforms = get_lag_transforms(feature_list = features['lag_transforms'])
    date_features = get_date_features(feature_list = features['date'])
    static_features = features['static']

    data_feat = MLForecast(
        models = [],
        freq = freq, 
        lags = lags,
        lag_transforms = lag_transforms,
        date_features = date_features
    ).preprocess(df = data, static_features = static_features)

    if remove_static:
        data_feat = data_feat.drop(static_features, axis = 1)

    return data_feat
