plot_anomalies <- function(data, anomaly_column = 'is_real_anomaly') {

    x_lab <- 'Date'
    y_lab <- unique(data$unique_id)
    data_anomalies <- data |> dplyr::filter(.data[[anomaly_column]] == 1)

    g <- data |> 
        ggplot2::ggplot(ggplot2::aes(x = .data[['ds']], y = .data[['y']])) +
        ggplot2::geom_line()

    g <- g +
        ggplot2::geom_point(data = data_anomalies, col = 'red', size = 2)

    g <- g +
        ggplot2::labs(
            x = x_lab,
            y = y_lab
        )
    
    return(g)

}

zscore_anomalies <- function(y, q = 3) {

    logging::loginfo('Zscore anomaly detection...')
    z <- (y - mean(y, na.rm = TRUE)) / sd(y, na.rm = TRUE)
    res <- abs(z) > q
    return(res)

}

tukey_anomaly <- function(y, extreme = FALSE) {

    logging::loginfo('Tukey anomaly detection...')
    n <- length(y)
    q1 <- quantile(y, 0.25, na.rm = TRUE)
    q3 <- quantile(y, 0.75, na.rm = TRUE)
    threshold <- (1.5 + 1.5 * extreme) * (q3 - q1)
    res <- y > q3 + threshold | y < q1 - threshold
    return(res)

}

barbato_anomaly <- function(y, extreme = FALSE) {

    logging::loginfo('Barbato anomaly detection...')
    n <- length(y)
    q1 <- quantile(y, 0.25, na.rm = TRUE)
    q3 <- quantile(y, 0.75, na.rm = TRUE)
    threshold <- (1.5 + 1.5 * extreme) * (q3 - q1) * (1 + log(n / 10))
    res <- y > q3 + threshold | y < q1 - threshold
    return(res)

}

anomaly_score <- function(anomaly_data, weights = NULL) {
	
	if (is.null(weights)) {
		logging::loginfo('Computing anomaly score...')
		score <- rowSums(anomaly_data) / ncol(anomaly_data)
	} else {
		logging::loginfo('Computing anomaly weighted score...')
        logging::loginfo(paste0('Weights = {', paste0(weights, collapse = ', '), '}'))
        if (sum(weights) != 1) {
            logging::logerror('Weights do not sum up to 1.')
        } else {
            score <- rowSums(purrr::map2_df(anomaly_data, weights, ~ .x * .y))
        }
		
	}
	return(score)
	
}

anomaly_select <- function(score, threshold = 0.5) {
	
	logging::loginfo('Selecting anomalies based on anomaly score...')
    logging::loginfo(paste0('Threshold = ', threshold))
    is_anomaly <- as.integer(score > threshold)
	return(is_anomaly)
	
}

ensemble_anomalies <- function(data, methods, ensemble_type = 'voting', threshold = 0.5, weights = NULL) {

    logging::loginfo('Estimating ensemble anomalies...')
    logging::loginfo(paste0(' [ Method: ', ensemble_type, ' ] '))
    anomaly_data <- data |> dplyr::select(dplyr::any_of(methods))

    if (ensemble_type == 'voting') {
        
        is_anomaly <- anomaly_data |> 
            anomaly_score(weights = NULL) |> 
            anomaly_select(threshold = threshold)
        res_data <- data |> dplyr::mutate('voting' = is_anomaly)

    } else if (ensemble_type == 'weighted_voting') {

        is_anomaly <- anomaly_data |> 
            anomaly_score(weights = weights) |> 
            anomaly_select(threshold = threshold)
        res_data <- data |> dplyr::mutate('weighted_voting' = is_anomaly)

    } else {
        logging::logerror('Unknown method ensemble_type.')
    }

    return(res_data)

}