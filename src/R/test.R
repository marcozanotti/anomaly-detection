reticulate::use_condaenv('global_retrain')

reticulate::source_python('src/Python/utils/utilities.py')
source('src/R/utils.R')
source('src/R/anomaly_detection.R')

library(tidyverse)
library(weird)


# Load data
data = get_data(path_list = c('data', 'nab'), name_list = c('nab', 'prep'))
data = data |> filter(unique_id == "speed_7578")

plot_anomalies(data, anomaly_column = 'is_real_anomaly')



# Statistical Tests
zscore_anomalies(data$y, q = 3)
weird::chauvenet_anomalies(data$y)
weird::peirce_anomalies(data$y)
weird::dixon_anomalies(data$y, alpha = 0.05)
weird::grubbs_anomalies(data$y, alpha = 0.05)



# Boxplots
tukey_anomaly(data$y, extreme = TRUE)
barbato_anomaly(data$y, extreme = TRUE)



# Density-based Methods

# specific distribution
weird::surprisals(
    data$y, 
    distribution = distributional::dist_normal(), 
    approximation = 'empirical',
    loo = TRUE
)
data |>
    dplyr::mutate(
        prob = weird::surprisals(
            data$y, distribution = distributional::dist_normal(), 
            approximation = 'empirical', loo = TRUE
        )
    ) |> 
    ggplot(aes(x = ds, y = y, color = prob < 0.01)) +
    geom_jitter(height = 0, width = 0.2) +
    scale_y_log10()

# linear regression
fit_lm <- lm(y ~ as.numeric(ds), data = data)
weird::surprisals(fit_lm, approximation = 'empirical', loo = TRUE)
data |>
    dplyr::mutate(prob = weird::surprisals(fit_lm, approximation = 'empirical', loo = TRUE)) |> 
    ggplot(aes(x = ds, y = y, color = prob < 0.01)) +
    geom_jitter(height = 0, width = 0.2) +
    scale_y_log10()

# gam regression
fit_gam <- mgcv::gam(y ~ as.numeric(ds), data = data)
weird::surprisals(fit_gam, approximation = 'empirical')
data |>
    dplyr::mutate(prob = weird::surprisals(fit_gam, approximation = 'empirical')) |> 
    ggplot(aes(x = ds, y = y, color = prob < 0.01)) +
    geom_jitter(height = 0, width = 0.2) +
    scale_y_log10()

# KDE
weird::surprisals(data$y, approximation = 'empirical', loo =  TRUE)
data |>
    dplyr::mutate(prob = weird::surprisals(data$y, approximation = 'empirical', loo =  TRUE)) |> 
    ggplot(aes(x = ds, y = y, color = prob < 0.01)) +
    geom_jitter(height = 0, width = 0.2) +
    scale_y_log10()
# weird::gg_hdrboxplot(
#     data, var1 = 'y', show_points = TRUE,
#     show_anomalies = TRUE
# )

weird::lookout_prob(data$y)
data |>
    dplyr::mutate(prob = weird::lookout_prob(data$y)) |> 
    ggplot(aes(x = ds, y = y, color = prob < 0.13)) +
    geom_jitter(height = 0, width = 0.2) +
    scale_y_log10()


# Distance-based Methods
weird::lof_scores(data$y)
weird::lof_scores(data$y, k = 10) > 1 # outliers if greater than 1
data |>
    dplyr::mutate(prob = weird::lof_scores(data$y, k = 5)) |> 
    ggplot(aes(x = ds, y = y, color = prob > 1)) +
    geom_jitter(height = 0, width = 0.2) +
    scale_y_log10()

weird::stray_scores(data$y)
weird::stray_anomalies(data$y)
data |>
    dplyr::mutate(prob = weird::stray_scores(data$y)) |> 
    ggplot(aes(x = ds, y = y, color = prob > 0.01)) +
    geom_jitter(height = 0, width = 0.2) +
    scale_y_log10()
data |>
    dplyr::mutate(is_anomaly = weird::stray_anomalies(data$y)) |> 
    plot_anomalies(anomaly_column = 'is_anomaly')



# Combine
data <- data |> 
    dplyr::mutate(
        "zscore" = as.integer(zscore_anomalies(data$y, q = 3)),
        "peirce" = as.integer(weird::peirce_anomalies(data$y)),
        "chauvenet" = as.integer(weird::chauvenet_anomalies(data$y)),
        "grubbs" = as.integer(weird::grubbs_anomalies(data$y, alpha = 0.05)),
        "dixon" = as.integer(weird::dixon_anomalies(data$y, alpha = 0.05)),
        "tukey" = as.integer(tukey_anomaly(data$y, extreme = TRUE)),
        "barbato" = as.integer(barbato_anomaly(data$y, extreme = TRUE))
    )
str(data)

methods <- c('zscore', 'peirce', 'chauvenet', 'grubbs', 'dixon', 'tukey', 'barbato')
data |> 
    ensemble_anomalies(
        methods = methods, 
        ensemble_type = 'voting',
        threshold = 0.5,
        weights = NULL
    ) |> 
    plot_anomalies(anomaly_column = 'voting')

methods <- c('zscore', 'peirce', 'chauvenet')
data |> 
    ensemble_anomalies(
        methods = methods, 
        ensemble_type = 'voting',
        threshold = 0.5,
        weights = NULL
    ) |> 
    plot_anomalies(anomaly_column = 'voting')

methods <- c('zscore', 'peirce', 'chauvenet')
data |> 
    ensemble_anomalies(
        methods = methods, 
        ensemble_type = 'weighted_voting',
        threshold = 0.5,
        weights = c(1, 0, 0)
    ) |> 
    plot_anomalies(anomaly_column = 'weighted_voting')
