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

    z <- (y - mean(y, na.rm = TRUE)) / sd(y, na.rm = TRUE)
    res <- abs(z) > q
    return(res)

}

tukey_anomaly <- function(y, extreme = FALSE) {

    n <- length(y)
    q1 <- quantile(y, 0.25, na.rm = TRUE)
    q3 <- quantile(y, 0.75, na.rm = TRUE)
    threshold <- (1.5 + 1.5 * extreme) * (q3 - q1)
    res <- y > q3 + threshold | y < q1 - threshold
    return(res)
  
}

barbato_anomaly <- function(y, extreme = FALSE) {

    n <- length(y)
    q1 <- quantile(y, 0.25, na.rm = TRUE)
    q3 <- quantile(y, 0.75, na.rm = TRUE)
    threshold <- (1.5 + 1.5 * extreme) * (q3 - q1) * (1 + log(n / 10))
    res <- y > q3 + threshold | y < q1 - threshold
    return(res)
  
}