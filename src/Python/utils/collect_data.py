
import sys
sys.path.insert(0, 'src/Python/utils')
import numpy as np
import pandas as pd
import pandas_flavor as pf
from utilities import save_data, load_data, get_frequency, get_dataset_frequency

import logging
module_logger = logging.getLogger('collect_data')


def download_data(dataset_name, save = True, ext = '.parquet'):

    """Function to download and save different time series datasets.

    Args:
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        save (bool, optional): Whether to save data or not. Train and test detasets
        are saved in data/_dataset_name/ as .parquet files. Defaults to True.
        ext (string, optional): File extension (default is '.parquet').

    Returns:
        pd.DataFrame: training and test dataframes.
    """

    if dataset_name == 'm5':
        
        module_logger.info(f'Downloading {dataset_name} train dataset...')
        train_df = pd.read_parquet('https://m5-benchmarks.s3.amazonaws.com/data/train/target.parquet') \
            .rename(columns = {'item_id': 'unique_id', 'timestamp': 'ds', 'demand': 'y'})
        module_logger.info(f'Downloading {dataset_name} test dataset...')
        test_df = pd.read_parquet('https://m5-benchmarks.s3.amazonaws.com/data/test/target.parquet') \
            .rename(columns = {'item_id': 'unique_id', 'timestamp': 'ds', 'demand': 'y'})
    
    elif dataset_name == 'vn1':

        module_logger.info(f'Downloading {dataset_name} train dataset...')
        # Phase 0 datset
        train_df0 = pd.read_csv('data/vn1/Phase 0 - Sales.csv')
        train_df0['unique_id'] = train_df0['Client'].astype(str) + '_' \
            + train_df0['Warehouse'].astype(str) + '_' \
            + train_df0['Product'].astype(str)
        train_df0.drop(columns = ['Client', 'Warehouse', 'Product'], axis = 1, inplace = True)
        # Phase 1 dataset
        train_df1 = pd.read_csv('data/vn1/Phase 1 - Sales.csv')
        train_df1['unique_id'] = train_df1['Client'].astype(str) + '_' \
            + train_df1['Warehouse'].astype(str) + '_' \
            + train_df1['Product'].astype(str)
        train_df1.drop(columns = ['Client', 'Warehouse', 'Product'], axis = 1, inplace = True)
        train_df = train_df0 \
            .merge(train_df1, how = 'left', on = 'unique_id') \
            .melt(id_vars = 'unique_id', var_name = 'ds', value_name  = 'y')

        module_logger.info(f'Downloading {dataset_name} test dataset...')
        test_df = pd.read_csv('data/vn1/Phase 2 - Sales.csv')
        test_df['unique_id'] = test_df['Client'].astype(str) + '_' \
            + test_df['Warehouse'].astype(str) + '_' \
            + test_df['Product'].astype(str)
        test_df.drop(columns = ['Client', 'Warehouse', 'Product'], axis = 1, inplace = True)
        test_df = test_df.melt(id_vars = 'unique_id', var_name = 'ds', value_name  = 'y')

    elif dataset_name == 'm4':

        # if other datasets are needed (like Hourly, Weekly, etc), add all here
        # creating a single train and test df
        
        module_logger.info(f'Downloading {dataset_name} train dataset...')      
        train_df = pd.read_csv('data/m4/Daily-train.csv')
        train_df.columns = ['unique_id'] + list(range(1, train_df.shape[1]))
        train_df = pd.melt(train_df, id_vars = ['unique_id'], var_name = 'ds', value_name = 'y')
        train_df = train_df.dropna()
        train_df['ds'] = train_df['ds'].astype('int')
        train_df = train_df.sort_values(['unique_id', 'ds']).reset_index(drop = True)

        module_logger.info(f'Downloading {dataset_name} test dataset...')
        test_df = pd.read_csv('data/m4/Daily-test.csv')
        test_df.columns = ['unique_id'] + list(range(1, test_df.shape[1]))
        test_df = pd.melt(test_df, id_vars = ['unique_id'], var_name = 'ds', value_name = 'y')
        test_df = test_df.dropna()
        test_df['ds'] = test_df['ds'].astype('int')
        len_train = train_df.groupby('unique_id').agg({'ds': 'max'}).reset_index()
        len_train.columns = ['unique_id', 'len_serie']
        test_df = test_df.merge(len_train, on = ['unique_id'])
        test_df['ds'] = test_df['ds'] + test_df['len_serie']
        test_df.drop('len_serie', axis = 1, inplace = True)
        test_df = test_df.sort_values(['unique_id', 'ds']).reset_index(drop = True)
    
    else:

        raise(f'Unknown dataset {dataset_name}')

    if save:
        save_data(
            data = train_df, 
            path_list = ['data', dataset_name], 
            name_list = [dataset_name, 'train'],
            ext = ext
        )
        save_data(
            data = test_df, 
            path_list = ['data', dataset_name], 
            name_list = [dataset_name, 'test'],
            ext = ext
        ) 

    return train_df, test_df

@pf.register_dataframe_method
def combine_train_test(train_df, test_df):

    """Function to combine train and test dataframes.

    Args:
        train_df (pd.DataFrame): training data in the Nixtla's format.
        test_df (pd.DataFrame): test data in the Nixtla format.

    Returns:
        pd.DataFrame: combined train and test dataframes.
    """

    module_logger.info('Combining train and test data...')
    combined_df = pd.concat([train_df, test_df], axis = 0, ignore_index = True)
    combined_df.sort_values(by = ['unique_id', 'ds'], inplace = True)
    combined_df.reset_index(drop = True, inplace = True)

    return combined_df

@pf.register_dataframe_method
def aggregate_data_by_frequency(data, dataset_name, frequency, drop_firstlast = False):

    """Function to aggregate dataframes by frequency.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string): The frequency of the data (e.g., 'daily', 'weekly').
    
    Returns:
        pd.DataFrame: dataframe aggregated by frequency.
    """

    def drop_first_last(df):
        return df.iloc[1:-1]

    freq_list = get_frequency(frequency)
    freq = freq_list[0]
    freq_df = get_frequency(get_dataset_frequency(dataset_name))[0]

    if freq != freq_df:
        module_logger.info(f'Aggregating {dataset_name} dataset into {frequency} frequency...')

        if 'unique_id' in data.columns:
            data_agg = data \
                .set_index('ds') \
                .groupby(['unique_id', pd.Grouper(freq = freq_list[2])]) \
                .sum() \
                .reset_index()

            if drop_firstlast:
                data_agg = data_agg \
                    .groupby('unique_id') \
                    .apply(drop_first_last, include_groups = False) \
                    .reset_index() \
                    .drop(columns = ['level_1'], axis = 1)

        else:
            data_agg = data \
                .set_index('ds') \
                .groupby([pd.Grouper(freq = freq_list[2])]) \
                .sum() \
                .reset_index()

            if drop_firstlast:
                data_agg = data_agg \
                    .apply(drop_first_last)

    else:
        data_agg = data      

    return data_agg

@pf.register_dataframe_method
def remove_series(data, min_series_length):

    """Function to remove series from the data based on their length.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        min_series_length (int): Minimum length of series to be kept.    

    Returns:
        pd.DataFrame: dataframe with series removed.
    """

    series_length = data.groupby('unique_id')['y'].count()

    module_logger.info(f'Removing series shorter than {min_series_length} observaions...')
    remove_ids = series_length[series_length < min_series_length].index.tolist()
    res_df = data[~data['unique_id'].isin(remove_ids)]

    n_series = len(data['unique_id'].unique())
    n_series_final = len(res_df['unique_id'].unique())
    n_series_to_remove = n_series - n_series_final
    p_series_to_remove = n_series_to_remove / n_series * 100
    module_logger.info(f'Removed {n_series_to_remove} series out of {n_series} ({p_series_to_remove:.1f}%).')
    module_logger.info(f'The final dataset contains {n_series_final} series.')

    return res_df

@pf.register_dataframe_method
def filter_series(data, max_series_length):

    """Function to remove series from the data based on their length.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        max_series_length (int): Maximum length of series to be kept.    

    Returns:
        pd.DataFrame: dataframe with series removed.
    """

    module_logger.info(f'Filtering series longer than {max_series_length} observaions...')
    res_df = data \
        .sort_values(['unique_id', 'ds']) \
        .groupby('unique_id') \
        .tail(n = max_series_length) \
        .reset_index(drop = True)

    n_obs = len(data)
    n_obs_final = len(res_df)
    n_obs_to_remove = n_obs - n_obs_final
    p_obs_to_remove = n_obs_to_remove / n_obs * 100
    module_logger.info(f'Removed {n_obs_to_remove} observations out of {n_obs} ({p_obs_to_remove:.1f}%).')
    module_logger.info(f'The final dataset contains {n_obs_final} observations.')

    return res_df

@pf.register_dataframe_method
def get_static_features(data, dataset_name):

    """Function to add static features to the data.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
    
    Returns:
        pd.DataFrame: dataframe with static features added.
    """

    module_logger.info(f'Extracting static features from {dataset_name} dataset...')

    # get splitted unique ids
    static_df = data['unique_id'] \
        .drop_duplicates() \
        .apply(lambda x: pd.Series(str(x).split("_"))) \
        .reset_index(drop = True)
    
    if dataset_name == 'm5':

        static_df['unique_id'] = static_df[0] + "_" \
            + static_df[1] + "_" \
            + static_df[2] + "_" \
            + static_df[3] + "_" \
            + static_df[4]
        static_df['item_id'] = static_df[0] + "_" + static_df[1] + "_" + static_df[2]
        static_df['item_id'] = static_df['item_id'].astype('category').cat.codes
        static_df['dept_id'] = static_df[0] + "_" + static_df[1]
        static_df['dept_id'] = static_df['dept_id'].astype('category').cat.codes
        static_df['cat_id'] = static_df[0]
        static_df['cat_id'] = static_df['cat_id'].astype('category').cat.codes
        static_df['store_id'] = static_df[3] + "_" + static_df[4]
        static_df['store_id'] = static_df['store_id'].astype('category').cat.codes
        static_df['state_id'] = static_df[3]
        static_df['state_id'] = static_df['state_id'].astype('category').cat.codes
        static_df.drop(columns = [0, 1, 2, 3, 4], axis = 1, inplace = True)

    elif dataset_name == 'vn1':
        static_df['unique_id'] = static_df[0] + "_" + static_df[1] + "_" + static_df[2]
        static_df['client'] = static_df[0].astype('category').cat.codes
        static_df['warehouse'] = static_df[1].astype('category').cat.codes
        static_df['product'] = static_df[2].astype('category').cat.codes
        static_df.drop(columns = [0, 1, 2], axis = 1, inplace = True)

    elif dataset_name == 'm4':
        static_df = pd.read_csv('data/m4/M4-info.csv')[['M4id', 'category']]
        static_df.columns = ['unique_id', 'category']
        static_df['category'] = static_df['category'].astype('category').cat.codes

    else:
        raise(f'Unknown dataset {dataset_name}')

    res_df = pd.merge(data, static_df, how = 'left', on = 'unique_id')

    return res_df

@pf.register_dataframe_method
def sampling_data(data, samples = 1000):

    """Function to sample dataframes.

    Args:
        data (pd.DataFrame): Input dataframe in Nixtla's format.
        samples (int, optional): Number of samples to be taken. Defaults to 1000.
    
    Returns:
        pd.DataFrame: sampled dataframe.
    """

    module_logger.info(f'Sampling {samples} series from data...')
    ids = data['unique_id'].unique()
    sample_ids = np.random.choice(ids, size = samples, replace = False)
    res_df = data[data['unique_id'] \
        .isin(sample_ids)] \
        .reset_index(drop = True)

    return res_df

def get_xregs_data(path_list, name_list, dataset_name, frequency, ext = '.parquet'):

    """Function to get external regressors (xregs) for the specified dataset.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of file names for external regressors.
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string): The frequency of the data (e.g., 'daily', 'weekly').
        ext (string, optional): Extension of the external regressors files. Defaults to '.parquet'.
    
    Returns:
        pd.DataFrame: dataframe with external regressors.
    """

    module_logger.info(f'Extracting external regressors for {dataset_name} dataset...')
    
    if dataset_name == 'm5':

        ext = '.csv'
        xreg_df = load_data(
            path_list = path_list, 
            name_list = name_list, 
            ext = ext
        )
        xreg_df['event'] = np.where(xreg_df['event_name_1'].isna(), 0, 1)
        xreg_df['event'] = xreg_df['event'].astype(int)
        xreg_df['date'] = pd.to_datetime(xreg_df['date'])
        xreg_df.rename(columns = {'date': 'ds'}, inplace = True)
        xreg_df = xreg_df[['ds', 'event']]
        xreg_df = aggregate_data_by_frequency(xreg_df, dataset_name, frequency, drop_firstlast = False)
        xreg_df['event'] = np.where(xreg_df['event'] == 0, 0, 1)

    elif dataset_name == 'vn1':

        raise ValueError(f'Xregs are not available for dataset {dataset_name}.')

    else:
        raise ValueError(f'Unknown dataset {dataset_name}.')

    return xreg_df

def prepare_data(dataset_name, frequency, static_features = True, xregs = True, save = True, ext = '.parquet'):

    """Function to prepare saved datasets.

    Args:
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        frequency (string, optional): The frequency of the data (e.g., 'daily', 'weekly').
        static_features (bool, optional): Whether to include static features. Defaults to True.
        xregs (bool, optional): Whether to include external regressors. Defaults to True.
        save (bool, optional): Whether to save the processed dataset. Defaults to False.
        ext (string, optional): Extension of the saved files. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: full dataframe.
    """

    train_df = load_data(
        path_list = ['data', dataset_name], 
        name_list = [dataset_name, 'train'], 
        ext = ext
    )
    train_df['unique_id'] = train_df['unique_id'].astype(str)

    test_df = load_data(
        path_list = ['data', dataset_name], 
        name_list = [dataset_name, 'test'],
        ext = ext
    )
    test_df['unique_id'] = test_df['unique_id'].astype(str)

    if dataset_name != 'm4':
        train_df['ds'] = pd.to_datetime(train_df['ds'])
        test_df['ds'] = pd.to_datetime(test_df['ds'])
    else:
        train_df['ds'] = train_df['ds'].astype('int')
        test_df['ds'] = test_df['ds'].astype('int')

    res_df = combine_train_test(train_df, test_df)
    del train_df, test_df

    if dataset_name != 'm4':
        res_df = aggregate_data_by_frequency(res_df, dataset_name, frequency, drop_firstlast = True)
    else:
        res_df = res_df[res_df['unique_id'].str.contains(get_frequency(frequency)[0])]
        
    if static_features:
        res_df = get_static_features(res_df, dataset_name)

    if xregs:
        xreg_df = get_xregs_data(
            path_list = ['data', dataset_name],  
            name_list = [dataset_name, 'xregs'], 
            dataset_name = dataset_name,
            frequency = frequency, 
            ext = ext
        )
        res_df = pd.merge(res_df, xreg_df, how = 'left', on = 'ds')

    if save:
        save_data(
            data = res_df, 
            path_list = ['data', dataset_name], 
            name_list = [dataset_name, frequency, 'prep'],
            ext = ext
        )

    return res_df

def get_data(path_list, name_list, min_series_length = None, max_series_length = None, samples = None, ext = '.parquet'):

    """Function to get the data.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of names to be used to create the file name.
        min_series_length (int, optional): Minimum length of series to be included. 
        Defaults to None.
        max_series_length (int, optional): Maximum length of series to be included. 
        Defaults to None.
        samples (int, optional): Number of samples to be included. Defaults to None.
        ext (string, optional): File extension (default is '.parquet').

    Returns:
        pd.Dataframe: The data.
    """

    res_df = load_data(path_list, name_list, ext)

    if min_series_length is not None:
        res_df = remove_series(res_df, min_series_length)

    if max_series_length is not None:
        res_df = filter_series(res_df, max_series_length)

    if samples is not None:
        res_df = sampling_data(res_df, samples)

    res_df.sort_values(by = ['unique_id', 'ds'], inplace = True)  

    return res_df
