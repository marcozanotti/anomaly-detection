
import sys
sys.path.insert(0, 'src/Python/utils')
import requests
import json
import numpy as np
import pandas as pd
import pandas_flavor as pf
from utilities import save_data, load_data, create_file_path, get_frequency, get_dataset_frequency

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

    if dataset_name == 'nab':
        
        module_logger.info(f'Downloading {dataset_name} datasets...')

        path = 'https://github.com/numenta/NAB/blob/master/data'
        file_ext = '.csv'
        file_names = {
            'artificialNoAnomaly': [
                'art_daily_no_noise', 
                'art_daily_perfect_square_wave',
                'art_daily_small_noise',
                'art_flatline',
                'art_noisy'
            ],
            'artificialWithAnomaly': [
                'art_daily_flatmiddle',
                'art_daily_jumpsdown',
                'art_daily_jumpsup',
                'art_daily_nojump',
                'art_increase_spike_density',
                'art_load_balancer_spikes'
            ],
            'realAWSCloudwatch': [
                'ec2_cpu_utilization_24ae8d',
                'ec2_cpu_utilization_53ea38',
                'ec2_cpu_utilization_5f5533',
                'ec2_cpu_utilization_77c1ca',
                'ec2_cpu_utilization_825cc2',
                'ec2_cpu_utilization_ac20cd',
                'ec2_cpu_utilization_c6585a',
                'ec2_cpu_utilization_fe7f93',
                'ec2_disk_write_bytes_1ef3de',
                'ec2_disk_write_bytes_c0d644',
                'ec2_network_in_257a54',
                'ec2_network_in_5abac7',
                'elb_request_count_8c0756',
                'grok_asg_anomaly',
                'iio_us-east-1_i-a2eb1cd9_NetworkIn',
                'rds_cpu_utilization_cc0c53',
                'rds_cpu_utilization_e47b3b'
            ], 
            'realAdExchange': [
                'exchange-2_cpc_results',
                'exchange-2_cpm_results',
                'exchange-3_cpc_results',
                'exchange-3_cpm_results',
                'exchange-4_cpc_results',
                'exchange-4_cpm_results'
            ],
            'realKnownCause': [
                'ambient_temperature_system_failure',
                'cpu_utilization_asg_misconfiguration',
                'ec2_request_latency_system_failure',
                'machine_temperature_system_failure',
                'nyc_taxi',
                'rogue_agent_key_hold',
                'rogue_agent_key_updown'
            ],
            'realTraffic': [
                'TravelTime_387',
                'TravelTime_451',
                'occupancy_6005',
                'occupancy_t4013',
                'speed_6005',
                'speed_7578',
                'speed_t4013'
            ],
            'realTweets': [
                'Twitter_volume_AAPL',
                'Twitter_volume_AMZN',
                'Twitter_volume_CRM',
                'Twitter_volume_CVS',
                'Twitter_volume_FB',
                'Twitter_volume_GOOG',
                'Twitter_volume_IBM',
                'Twitter_volume_KO',
                'Twitter_volume_PFE',
                'Twitter_volume_UPS'
            ]
        }
        i = 1

        # labels
        module_logger.info('Downloading labels datasets...')
        labels_url = 'https://github.com/numenta/NAB/blob/master/labels/combined_labels.json?raw=true'
        resp = requests.get(labels_url)
        labels = json.loads(resp.text)

        for k in list(file_names.keys()):

            module_logger.info(f'Directory {k}...')
            fnm_tmp = file_names[k]
            
            for f in fnm_tmp:

                module_logger.info(f'File {f}...')
                fext_tmp = f + file_ext
                fpath_tmp = create_file_path([path, k, fext_tmp], end = '')
                gh_path_tmp = fpath_tmp + '?raw=true'
                lbl_tmp = labels[k + '/' + fext_tmp]

                df_tmp = pd.read_csv(gh_path_tmp) \
                    .rename(columns = {'timestamp': 'ds', 'value': 'y'}) \
                    .assign(unique_id = f, is_real_anomaly = 0)
                df_tmp = df_tmp[['unique_id', 'ds', 'y', 'is_real_anomaly']]
                df_tmp.loc[df_tmp['ds'].isin(lbl_tmp), 'is_real_anomaly'] = 1

                if save:
                    save_data(
                        data = df_tmp, 
                        path_list = ['data', dataset_name], 
                        name_list = [dataset_name, f],
                        ext = ext
                    )

                i += 1                

    else:

        raise(f'Unknown dataset {dataset_name}')

    module_logger.info(f'Done! Downloaded {i} files.')

    return

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

def prepare_data(dataset_name, save = True, ext = '.parquet'):

    """Function to prepare saved datasets.

    Args:
        dataset_name (string): Name of the dataset (e.g., 'm5', 'm4').
        save (bool, optional): Whether to save the processed dataset. Defaults to False.
        ext (string, optional): Extension of the saved files. Defaults to '.parquet'.

    Returns:
        pd.DataFrame: full dataframe.
    """

    df = load_data(
        path_list = ['data', dataset_name], 
        name_list = [dataset_name], 
        ext = ext
    )
    df['unique_id'] = df['unique_id'].astype(str)
    df['ds'] = pd.to_datetime(df['ds'])

    if save:
        save_data(
            data = df, 
            path_list = ['data', dataset_name], 
            name_list = [dataset_name, 'prep'],
            ext = ext
        )

    return df

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
