import numpy as np
from scipy.stats import norm
from scipy.stats import t
import pandas as pd
import pandas_flavor as pf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

    module_logger.info('Computing anomalies with Z-Score method...')

    data_anom = df.copy()
    y = data_anom['y']
    z = (y - np.nanmean(y)) / np.nanstd(y, ddof=1)
    is_anomaly = np.abs(z) > q
    data_anom['Zscore'] = is_anomaly.astype('int')

    return data_anom

@pf.register_dataframe_method
def Pierce(df):

    """
    Peirce anomaly detection applied to a DataFrame column.

    Args:
        df (pd.DataFrame): Input dataframe with column 'y'.

    Returns:
        pd.DataFrame: Original dataframe with one extra int column (1 = anomaly, 0 = normal).
    """

    logging.info('Computing anomalies with Peirce method...')

    def peirce_threshold(n):
        module_logger.info('Computing Pierce threshold...')
        # Check we have enough observations
        if (n - 2) <= 0:
            return np.nan
        # Initialize
        x = 1.0
        oldx = np.inf
        root2pie = np.sqrt(2 / np.pi / np.e)
        # Eq (B) after taking logs
        LnQN = (n - 1) * np.log(n - 1) - n * np.log(n)
        # Loop until convergence
        while abs(x - oldx) >= n * np.finfo(float).eps:
            # Eq (D)
            R1 = 2 * np.exp(0.5 * (x**2 - 1)) * norm.sf(x)  # sf = 1 - cdf
            # Eq (A') after taking logs and solving for R
            R2 = np.exp(LnQN - (n - 1) * 0.5 * np.log((n - 1 - x**2) / (n - 2)))
            # Derivatives wrt x
            R1d = x * R1 - root2pie
            R2d = x * (n - 1) / (n - 1 - x**2) * R2
            # Update x
            oldx = x
            x = oldx - (R1 - R2) / (R1d - R2d)
        return x

    data_anom = df.copy()
    y = df['y']
    z = (y - np.nanmean(y)) / np.nanstd(y, ddof=1)
    threshold = peirce_threshold(len(y))
    is_anomaly = np.abs(z) > threshold if not np.isnan(threshold) else np.zeros_like(z, dtype = bool)
    data_anom['Pierce'] = is_anomaly.astype(int)

    return data_anom

@pf.register_dataframe_method
def Chauvenet(df):

    """
    Chauvenet anomaly detection applied to a DataFrame column.

    Args:
        df (pd.DataFrame): Input dataframe with column 'y'.

    Returns:
        pd.DataFrame: Original dataframe with one extra int column (1 = anomaly, 0 = normal).
    """

    logging.info('Computing anomalies with Chauvenet method...')

    data_anom = df.copy()
    y = df['y']
    z = (y - np.nanmean(y)) / np.nanstd(y, ddof=1)
    threshold = norm.ppf(1 - 0.25 / len(y))
    is_anomaly = np.abs(z) > threshold
    data_anom['Chauvenet'] = is_anomaly.astype(int)

    return data_anom

@pf.register_dataframe_method
def Grubbs(df, alpha = 0.05):

    """
    Grubbs anomaly detection applied to a DataFrame column.

    Args:
        df (pd.DataFrame): Input dataframe with column 'y'.
        alpha (float): Significance level (default 0.05).

    Returns:
        pd.DataFrame: Original dataframe with one extra int column 'Grubbs' (1 = anomaly, 0 = normal).
    """

    logging.info('Computing anomalies with Grubbs method...')

    data_anom = df.copy()
    y = df['y']
    n = len(y)
    z = (y - np.nanmean(y)) / np.nanstd(y, ddof = 1)
    t2 = t.ppf(1 - alpha / (2 * n), df = n - 2)
    threshold = (n - 1) / np.sqrt(n) * np.sqrt(t2**2 / (n - 2 + t2**2))
    is_anomaly = np.abs(z) > threshold
    data_anom['Grubbs'] = is_anomaly.astype(int)

    return data_anom

@pf.register_dataframe_method
def Dixon(df, alpha = 0.05, two_sided = True):

    """
    Dixon anomaly detection applied to a DataFrame column.

    Args:
        df (pd.DataFrame): Input dataframe with column 'y'.
        alpha (float): Significance level.
        two_sided (bool): If True, test both minimum and maximum; else only maximum.

    Returns:
        pd.DataFrame: Original dataframe with one extra int column 'Dixon' (1 = anomaly, 0 = normal).
    """

    logging.info('Computing anomalies with Dixon method...')

    data_anom = df.copy()
    y = df['y']
    n = len(y)

    sorty = np.sort(y)
    min_idx = np.argmin(y)
    max_idx = np.argmax(y)

    # Compute Q statistic
    if two_sided:
        Q = max(sorty[1] - sorty[0], sorty[-1] - sorty[-2]) / (sorty[-1] - sorty[0])
    else:
        Q = (sorty[-1] - sorty[-2]) / (sorty[-1] - sorty[0])
        alpha *= 2

    # --- Critical value interpolation ---
    # Functions for logit and loglog transforms
    def logit(u):
        return np.log(u / (1 - u))

    def loglog(u):
        return np.log(np.log(u))

    # Subset alpha values
    dixon_cv = pd.read_csv('data/dixon_cv.csv')
    alpha_grid = np.sort(dixon_cv['alpha'].unique())
    nearest_alpha = alpha_grid[np.argsort(np.abs(logit(alpha_grid) - logit(alpha)))][:4]

    # Subset n values
    alpha_only_model = n in range(3, 51)
    if alpha_only_model:
        nearest_n = [n]
    else:
        n_grid = np.sort(dixon_cv['n'].unique())
        nearest_n = n_grid[np.argsort(np.abs(loglog(n_grid) - loglog(n)))][:4]

    cv_subset = dixon_cv[
        (dixon_cv['alpha'].isin(nearest_alpha)) & (dixon_cv['n'].isin(nearest_n))
    ].copy()

    cv_subset['loglogn'] = loglog(cv_subset['n'])
    cv_subset['logitalpha'] = logit(cv_subset['alpha'])
    cv_subset['logcv'] = np.log(cv_subset['cv'])

    # Fit regression
    if alpha_only_model:
        X = PolynomialFeatures(3, include_bias=False).fit_transform(cv_subset[['logitalpha']])
    else:
        poly_alpha = PolynomialFeatures(2, include_bias=False).fit_transform(cv_subset[['logitalpha']])
        poly_n = PolynomialFeatures(2, include_bias=False).fit_transform(cv_subset[['loglogn']])
        X = np.hstack([poly_alpha, poly_n, (cv_subset['logitalpha'] * cv_subset['loglogn']).values.reshape(-1, 1)])

    y_cv = cv_subset['logcv'].values
    model = LinearRegression().fit(X, y_cv)

    # Predict threshold
    if alpha_only_model:
        X_pred = PolynomialFeatures(3, include_bias=False).fit_transform(np.array([[logit(alpha)]]))
    else:
        poly_alpha_pred = PolynomialFeatures(2, include_bias=False).fit_transform(np.array([[logit(alpha)]]))
        poly_n_pred = PolynomialFeatures(2, include_bias=False).fit_transform(np.array([[loglog(n)]]))
        X_pred = np.hstack([poly_alpha_pred, poly_n_pred, (np.array([[logit(alpha) * loglog(n)]]))])

    threshold = np.exp(model.predict(X_pred)[0])

    # Determine anomalies
    output = np.zeros(n, dtype=int)
    if Q > threshold:
        if two_sided:
            if (sorty[1] - sorty[0]) / (sorty[-1] - sorty[0]) > threshold:
                output[min_idx] = 1
        if (sorty[-1] - sorty[-2]) / (sorty[-1] - sorty[0]) > threshold:
            output[max_idx] = 1

    data_anom['Dixon'] = output

    return data_anom

@pf.register_dataframe_method
def Tukey(df, extreme = False):

    """
    Tukey anomaly detection applied to a DataFrame column.

    Args:
        df (pd.DataFrame): Input dataframe.
        extreme (bool): If True, use more extreme threshold (3 * IQR instead of 1.5 * IQR).

    Returns:
        pd.DataFrame: Original dataframe with one extra boolean column.
    """

    logging.info('Computing anomalies with Tukey method...')
    
    data_anom = df.copy()
    y = df['y']
    q1 = np.nanpercentile(y, 25)
    q3 = np.nanpercentile(y, 75)
    threshold = (1.5 + 1.5 * int(extreme)) * (q3 - q1)
    is_anomaly = (y > q3 + threshold) | (y < q1 - threshold)
    data_anom['Tukey'] = is_anomaly.astype('int')

    return data_anom

@pf.register_dataframe_method
def Barbato(df, extreme = False):

    """
    Barbato anomaly detection applied to a DataFrame column.

    Args:
        df (pd.DataFrame): Input dataframe with column 'y'.
        extreme (bool): If True, use more extreme threshold 
        (3 * IQR instead of 1.5 * IQR).

    Returns:
        pd.DataFrame: Original dataframe with one extra int column 'Barbato'
        (1 = anomaly, 0 = normal).
    """

    logging.info('Computing anomalies with Barbato method...')

    data_anom = df.copy()
    y = df['y']
    n = len(y)
    q1 = np.nanpercentile(y, 25)
    q3 = np.nanpercentile(y, 75)
    threshold = (1.5 + 1.5 * int(extreme)) * (q3 - q1) * (1 + np.log(n / 10))
    is_anomaly = (y > q3 + threshold) | (y < q1 - threshold)
    data_anom['Barbato'] = is_anomaly.astype(int)

    return data_anom

@pf.register_dataframe_method
def Hampel(df, bandwidth, k = 3):

    """
    Hampel anomaly detection applied to a DataFrame column.

    Args:
        df (pd.DataFrame): Input dataframe with column 'y'.
        bandwidth (int): Size of the sliding window (must be integer).
        k (float): Threshold multiplier (default 3).

    Returns:
        pd.DataFrame: Original dataframe with one extra int column 'Hampel' (1 = anomaly, 0 = normal).
    """

    logging.info('Computing anomalies with Hampel method...')

    data_anom = df.copy()

    if abs(bandwidth - round(bandwidth)) > 1e-8:
        raise ValueError('Bandwidth must be an integer')

    bandwidth = int(round(bandwidth))
    y = data_anom['y']
    n = len(y)

    # Running median
    m = pd.Series(y).rolling(window = 2 * bandwidth + 1, center = True, min_periods = 1).median().to_numpy()
    diff = np.abs(y - m)

    # Running MAD, set endpoints to inf
    mad = np.full(n, np.inf)
    for i in range(bandwidth, n - bandwidth):
        window = y[i - bandwidth : i + bandwidth + 1]
        mad[i] = np.median(np.abs(window - m[i]))

    is_anomaly = diff > mad * k * 1.482602
    data_anom['Hampel'] = is_anomaly.astype(int)

    return data_anom
