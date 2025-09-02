
import os
import yaml
import pandas as pd
import pandas_flavor as pf
import pyarrow.parquet as pq
import logging
import logging.config
from time import gmtime, strftime

module_logger = logging.getLogger('utilities')


def get_config(config_file):
    """Function to load configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration as a dictionary.
    """

    with open(config_file, 'rt') as f:
        config = yaml.safe_load(f.read())

    return config

def configure_logging(config_file, name_list):

    """Function to configure logging.

    Args:
        config (dict): Logging configuration in YAML format.
        log_file_name (str): Name of the log file.
    """

    # grey = '\x1b[38;20m'
    yellow = '\x1b[33;20m'
    # red = '\x1b[31;20m'
    # bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    time_suffix = strftime("%Y%m%d_%H%M%S", gmtime())
    log_file = create_file_name(
        name_list = name_list + [time_suffix], 
        ext = '.log'
    )
    cfg = get_config(config_file)
    cfg['handlers']['file']['filename'] = 'logs/' + log_file   
    f = cfg['formatters']['colored']['format']
    cfg['formatters']['colored']['format'] = yellow + f + reset
    
    logging.config.dictConfig(cfg)

    return

def create_logger(name = None):

    """Function to create a logger.

    Returns:
        logging.Logger: Logger object.
    """

    return logging.getLogger(name)

def stop_logger(logger):

    """Function to stop logging.

    Args:
        logger (logging.Logger): Logger object.
    """

    logger.handlers.clear()
    logging.shutdown()

    return

def create_file_path(path_list, sep = '/', end = '/'):
    """Function to create a file path from a list of directories.

    Args:
        path_list (list): List of directories to be joined.
    
    Returns:
        string: The file path.
    """

    if len(path_list) == 1:
        path = path_list[0]
    else:
        path = sep.join(str(x) for x in path_list) + end

    return path

def create_file_name(name_list, ext = None):

    """Function to create a file name from a list of names.

    Args:
        name_list (list): List of names to be used to create the file name.
        ext (str, optional): Extension of the file. Defaults to '.parquet'.
    
    Returns:
        string: The file name.
    """
    if len(name_list) == 1:
        file_name = name_list[0]
    else:
        file_name = '_'.join(str(x) for x in name_list)
    
    if ext is not None:
        file_name += ext

    return file_name

def get_file_name(path_list, name_list = None, ext = '.parquet', remove_ext = True):

    """Function to get file names from a given path.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of names to be used to filter the files. Defaults to None.
        ext (str, optional): Extension of the files. Defaults to '.parquet'.
        remove_ext (bool, optional): Whether to remove the extension from the file names. 
        Defaults to True.
    
    Returns:
        list: List of file names.
    """

    path = create_file_path(path_list)
    file_names = os.listdir(path)
    if name_list is not None:
        for n in name_list:
            file_names = list(filter(lambda x: str(n) in x, file_names))
    if remove_ext:
        file_names = [s.replace(ext, "") for s in file_names]

    return file_names

def combine_and_save_files(
    path_list_to_read, path_list_to_write, name_list, ext = '.parquet', files_to_read = None
):

    """Function to combine and save multiple files into a single file.

    Args:
        path_list_to_read (list): List of directories to be joined for reading.
        path_list_to_write (list): List of directories to be joined for writing.
        name_list (list): List of names to be used to filter the files.
        ext (str, optional): Extension of the files. Defaults to '.parquet'.
        files_to_read (list, optional): List of specific files to read. Defaults to None.
    """

    module_logger.info('Combining and saving files...')

    if files_to_read is None:
        path_to_read = create_file_path(path_list_to_read)
        file_names = get_file_name(
            path_list = [path_to_read], 
            name_list = name_list,
            ext = ext, 
            remove_ext = False
        )
        files = [path_to_read + f for f in file_names]
    else:
        files = files_to_read

    # files.sort(key = lambda x: int("".join([i for i in x if i.isdigit()])))
    path_to_write = create_file_path(path_list_to_write)
    if not os.path.exists(path_to_write):
        os.makedirs(path_to_write)
    write_file_name = create_file_name(name_list = name_list, ext = ext) 

    if ext == '.parquet':
        if files:
            schema = pq.ParquetFile(files[0]).schema_arrow
            with pq.ParquetWriter(path_to_write + write_file_name, schema = schema) as writer:
                for f in files:
                    writer.write_table(pq.read_table(f, schema = schema))
    else:
        raise ValueError(f'Unsupported extension {ext}')

    return

def remove_file(path_list, name_list, ext = '.parquet'):

    """Function to remove files from a given path.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of names to be used to filter the files.
        ext (str, optional): Extension of the files. Defaults to '.parquet'.
    """

    module_logger.info('Removing files...')
    path_to_remove = create_file_path(path_list)
    files = get_file_name(
        path_list = [path_to_remove], name_list = name_list,
        ext = ext, remove_ext = False
    )
    for f in files:
        os.remove(f'{path_to_remove}{f}')

    return

@pf.register_dataframe_method
def save_data(data, path_list, name_list, ext = '.parquet'):

    """Function to save dataframes.

    Args:
        data (pd.DataFrame): Data to be saved.
        path_list (list): List of directories to be joined.
        name_list (list): List of names to be used to create the file name.
        ext (string, optional): File extension (default is '.parquet').
    """
    
    path = create_file_path(path_list)
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = create_file_name(name_list)
    module_logger.info(f'Saving {file_name} dataset...')

    if ext == '.parquet':
        data.to_parquet(f'{path}{file_name}{ext}')
    elif ext == '.csv':
        data.to_csv(f'{path}{file_name}{ext}')
    else:
        raise(f'Unsupported file extension {ext}. Only .parquet and .csv are allowed')

def load_data(path_list, name_list, ext = '.parquet'):

    """Function to load the data.

    Args:
        path_list (list): List of directories to be joined.
        name_list (list): List of names to be used to create the file name.
        ext (string, optional): File extension (default is '.parquet').

    Returns:
        pd.Dataframe: The data.
    """

    path = create_file_path(path_list)
    file_name = create_file_name(name_list)
    module_logger.info(f'Loading {file_name} dataset...')

    if ext == '.parquet':
        res_df = pd.read_parquet(f'{path}{file_name}{ext}')
    elif ext == '.csv':
        res_df = pd.read_csv(f'{path}{file_name}{ext}')
    else:
        raise(f'Unsupported file extension {ext}. Only .parquet and .csv are allowed')

    return res_df

def get_frequency(frequency):
    """Function to get the frequency of the dataset.

    Args:
        frequency (str): frequency of the dataset.
    
    Returns:
        str: frequency.
    """

    module_logger.info('Defining frequency...')

    if frequency == None:
        freq = [None, None, None]
    elif isinstance(frequency, int):
        freq = [frequency, frequency, frequency]
    elif isinstance(frequency, str):
        if frequency == 'hourly':
            freq = ['H', 24, 'h']
        elif frequency == 'daily':
            freq = ['D', 7, 'D']
        elif frequency == 'weekly':
            freq = ['W-MON', 52, 'W-MON']
        elif frequency == 'monthly':
            freq = ['ME', 12, 'ME']
        elif frequency == 'quarterly':
            freq = ['QE', 4, 'QE']
        elif frequency == 'yearly':
            freq = ['YE', 1, 'YE']
        else:
            raise ValueError(f'Invalid frequency: {frequency}')  
    else:
        raise ValueError(f'Invalid frequency: {frequency}')

    return freq

def get_dataset_frequency(dataset_name, series_name = None):

    """Function to get the frequency of a specific dataset.

    Args:
        dataset_name (str): name of the dataset.
        series_name (str): name of the time series.
    
    Returns:
        str: frequency.
    """

    module_logger.info(f'Getting {dataset_name} frequency...')

    if dataset_name == 'nab':
        freq = 'yearly'
    else:
        raise ValueError(f'Invalid dataset: {dataset_name}')

    return freq

def get_retrain_scenarios(frequency):
    """Function to get the retrain scenarios for a specific frequency.

    Args:
        frequency (str): frequency of the dataset.
    
    Returns:
        list: retrain scenarios.
    """

    module_logger.info('Getting retrain scenarios...')

    if frequency == 'daily':
        scn = [7, 14, 21, 30, 60, 90, 120, 150, 180, 364]
    elif frequency == 'weekly':
        scn = [1, 2, 3, 4, 6, 8, 10, 13, 26, 52]
    elif frequency == 'monthly':
        scn = [1, 2, 3, 4, 5, 6, 9, 12, 15, 18]
    else:
        raise ValueError(f'Invalid frequency: {frequency}')

    return scn

