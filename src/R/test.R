library(tidyverse)
remotes::install_github("robjhyndman/weird-package")


otsad::nyc_taxi |> str()
otsad::nyc_taxi$is.real.anomaly |> table()


plot(x = 1:nrow(otsad::nyc_taxi), y = otsad::nyc_taxi$value, type = 'l')

