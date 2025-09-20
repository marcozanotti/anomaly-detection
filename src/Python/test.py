import sys
sys.path.insert(0, 'src/Python/utils')
from utilities import (
    configure_logging, create_logger, stop_logger
)
from collect_data import get_data
from anomaly_models import Zscore, Pierce, Chauvenet, Dixon, Grubbs, Tukey, Barbato, Hampel  


configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = ['nab']
)
logger = create_logger()

data = get_data(['data', 'nab'], ['nab', 'prep'])
data = data.query('unique_id == "speed_7578"')

Zscore(data, q = 3)
Pierce(data)
Chauvenet(data)
Dixon(data, alpha = 0.05, two_sided = True)
Grubbs(data, alpha = 0.05)
Tukey(data, extreme = False)
Barbato(data, extreme = False)
Hampel(data, bandwidth = 3, k = 3)

stop_logger(logger)






import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from fit_anomaly_models import retrain_anomaly_model

config = get_config('config/fit/TEST_retrain_anom_nab_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        'train', 'anomaly'
    ]
)
logger = create_logger()

retrain_anomaly_model(config = config)

stop_logger(logger)







import sys
sys.path.insert(0, 'src/Python/utils')
import os
from utilities import (
    get_config, configure_logging, create_logger, stop_logger
)
from fit_anomaly_ensembles import fit_anomaly_ensembles

config = get_config('config/ensemble/TEST_ensemble_nab_config.yaml')
configure_logging(
    config_file = 'config/log_config.yaml', 
    name_list = [
        config['dataset']['dataset_name'], 
        'train', 'anomaly'
    ]
)
logger = create_logger()

fit_anomaly_ensembles(config = config)

stop_logger(logger)







import sys
sys.path.insert(0, 'src/Python/utils')
import numpy as np
import pandas as pd
from collect_data import get_data


data = get_data(['data', 'nab'], ['nab', 'prep'])

# 1. number of unique_id
n_unique_id = data['unique_id'].nunique()
print("Number of unique_id:", n_unique_id)

# 2. number of observations per unique_id
obs_per_id = data.groupby('unique_id').size().reset_index(name='n_obs')

# 3. min and max ds for each unique_id
min_max_ds = (
    data.groupby('unique_id')['ds'].agg(min_ds='min', max_ds='max').reset_index()
)

# 4. how many is_real_anomaly = 1 for each unique_id
anomalies_per_id = (
    data.groupby('unique_id')['is_real_anomaly'].sum().reset_index(name='n_anomalies')
)

# 5. ds of the first is_real_anomaly = 1 for each unique_id
first_anomaly_ds = (
    data.loc[data['is_real_anomaly'] == 1].groupby('unique_id')['ds'].min().reset_index(name='first_anomaly_ds')
)

# 6 & 7. observations before and after the first anomaly
before_after = []
for uid, g in data.groupby('unique_id'):
    n_total = len(g)
    if (g['is_real_anomaly'] == 1).any():
        first_ds = g.loc[g['is_real_anomaly'] == 1, 'ds'].min()
        before = np.round((g['ds'] < first_ds).sum(), 0)
        after = np.round((g['ds'] > first_ds).sum(), 0)
        before_rel = np.round(before / n_total, 2)
        after_rel = np.round(after / n_total, 2)
    else:
        first_ds, before, after, before_rel, after_rel = None, None, None, None, None
    before_after.append((uid, before, after, before_rel, after_rel))
before_after_df = pd.DataFrame(
    before_after,
    columns=[
        'unique_id',
        'n_before_first_anomaly',
        'n_after_first_anomaly',
        'rel_before_first_anomaly',
        'rel_after_first_anomaly'
    ]
)

# Merge everything into one summary table
summary = (
    obs_per_id
    .merge(min_max_ds, on='unique_id', how='left')
    .merge(anomalies_per_id, on='unique_id', how='left')
    .merge(first_anomaly_ds, on='unique_id', how='left')
    .merge(before_after_df, on='unique_id', how='left')
)

print(summary.head())  # show first rows
summary



import matplotlib.pyplot as plt

for uid, g in data.groupby('unique_id'):
    plt.figure(figsize=(12, 4))
    
    # Ensure sorted by time
    g = g.sort_values("ds")
    
    # Compute split point
    split_idx = int(len(g) * 0.3)
    
    # First 30% in orange
    plt.plot(g['ds'].iloc[:split_idx], g['y'].iloc[:split_idx],
             color='orange', label='First 30%')
    
    # Remaining in blue
    plt.plot(g['ds'].iloc[split_idx:], g['y'].iloc[split_idx:],
             color='blue', label='Remaining 70%')
    
    # Plot anomalies in red
    anomalies = g[g['is_real_anomaly'] == 1]
    if not anomalies.empty:
        plt.scatter(anomalies['ds'], anomalies['y'],
                    color='red', label='Anomaly', zorder=5)
    
    plt.title(f"Time series: {uid}")
    plt.xlabel("ds")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
