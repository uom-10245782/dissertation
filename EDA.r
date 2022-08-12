### EDA ###
# Exploring AURN, TFGM, ECMWF data
# Using R 
###########
 # Set wd
setwd("/.")

# Install necessary packages #
# Packages need to be installed only once

# corrplot package 
# source: https://cran.r-project.org/web/packages/corrplot/vignettes/corrplot-intro.html
install.packages("corrplot")
# openair library
# Source: https://rdrr.io/cran/openair/
install.packages("openair")
library(openair)
# correlation matrix plot
source("http://www.sthda.com/upload/rquery_cormat.r")
# pairplot
install.packages("ggplot2")            
install.packages("GGally")
library("ggplot2")                     
library("GGally")   

###########
# 1. Descriptive statistics #
  # a. All hourly data for 2020
# AURN (NO2, O3, SO2, NO, NOxNO2, PM25, PM10, wd, ws, temp)
# TfGM (traffic volume)
# ECMWF (msl)
 
# Read data as dataframe 
df1 <- read.csv(file='MAN_DATA_2020.csv')

#  Convert to date if not already
df1$Date <- as.Date(df1$date)
# Get time
df1$Time <- format(df1$date,"%H:%M:%S")
#  Get months
df1$Month <- months(df1$date)
#  Get years
df1$Year <- format(df1$date,format="%y")

# Collect summary statistics
write.csv(summary(df1), 'descriptive_statistics_summary.csv')
# Collect standard deviation 
apply(df1, 2, sd, na.rm=TRUE)

  # b. Explore NO2 
# Find % of time exceeding EU limit 40 ug/m3
nrow(df1[df1$NO2>40, ])
# %time = (nrow exceeding / total nrows)*100
(1915/8784)*100
# == 21.8%

# Aggregate 'NO2' on months and year 
# and collect statistics 
# mean
aggregate( NO2 ~ Month + Year , df1 , mean )
# max
aggregate( NO2 ~ Month + Year , df1 , max )
# min
aggregate( NO2 ~ Month + Year , df1 , min )

  # c. Wind (speed and direction)
# Find calm/turbulent wind %
# Calm <1mph
nrow(df1[df1$ws<1, ])
# = 471
(471/8784)*100
# %time == 5.36%

# Calm/light air <3mph %
nrow(df1[df1$ws<3, ])
# = 4010
(4010/8784)*100
# %time == 40.65%

# Find NO2 summary stastitics using wind direction 
# Use 45 degree bins
# eg. NE: 22.5 - 67.5; E: 67.5 - 112.5; etc.
# get mean
tapply(df1$NO2, cut(df1$wd, seq(22.5, 337.5, by=45)), mean, na.rm=TRUE)
# get max
tapply(df1$NO2, cut(df1$wd, seq(22.5, 337.5, by=45)), max, na.rm=TRUE)
# get min
tapply(df1$NO2, cut(df1$wd, seq(22.5, 337.5, by=45)), min, na.rm=TRUE)
# get N: 337.5 - 22.5 
df225 <- df1[df1$wd<22.5, ]
df3375 <- df1[df1$wd>337.5, ]
df2 <- rbind(df225, df3375)
mean(df2$NO2, na.rm=TRUE)
max(df2$NO2, na.rm=TRUE)
min(df2$NO2, na.rm=TRUE)


####################
2. Traffic, MSL 
####################
# 3. Visualisation #
  # a. NO2 

# Timeseries line #
plot.ts(df1$NO2, col=rgb(0.1,0.1,0.7,0.5))

# Autocorrelation #
# Install and load tseries library 
# Source: https://cran.r-project.org/web/packages/tseries/index.html
install.packages('tseries')
library(tseries)
# include pl=FALSE to show table // include pl=TRUE to show plot
# set lag to be # hours inspecting
# acf = autocorrelation function
# plot 96 hour lag (4 days) - random day
acf(df1$NO2, na.action = na.pass, lag=96, main='NO2 Autocorrelation', pl=TRUE) 
# plot 96 hour lag (4 days) - march and november 
# subset dataframe
df_nov <- df1[df1$Month == 'November', ]
df_mar <- df1[df1$Month == 'March', ]
# plot
par(mfrow=c(1,2))    # set the plotting area into a 1*2 array
ac_mar <- acf(df_mar$NO2, na.action = na.pass, lag=96, main='March', pl=FALSE) 
ac_nov <- acf(df_nov$NO2, na.action = na.pass, lag=96, main='November', pl=FALSE) 

# Partial Autocorrelation # 
# pacf = partial autocorrelation function
# plot 24 hour lag (1 day)
pacf(df1$NO2, na.action = na.pass, lag=24, main='Partial Autocorrelation', pl=TRUE) 

""" REVIEWWWWWWWWWWWW
  # b. Boxplot
# Using boxplot
# NO2 #
# BY Month
boxplot(df1$NO2 ~ reorder(format(df1$Date,'%b %y'),df1$Date), 
        outline = FALSE, col=rgb(0.1,0.1,0.7,0.5)) 

# BY Time of day
boxplot(df1$NO2 ~ reorder(format(df1$Time),df1$Time), 
        outline = FALSE, col='lightblue') 
# WS #
# BY month
boxplot(df1$ws ~ reorder(format(df1$Date,'%b %y'),df1$Date), 
        outline = FALSE, col='darkseagreen3') 

# Temp #
# BY month
boxplot(df1$temp ~ reorder(format(df1$Date,'%b %y'),df1$Date), 
        outline = FALSE, col='darkorange') 

############## OR ###############
# BY time of day #
par(mfrow=c(1,2))
# NO2 
boxplot(df1$NO2 ~ reorder(format(df1$Time),df1$Time), 
        outline = FALSE, col='lightblue',
        xlab="Time of Day",
        ylab="Concentration ug/m3",
        main = "NO2")
# traffic
boxplot(df1$Traffic ~ reorder(format(df1$Time),df1$Time), 
        outline = FALSE, col='darkorange',
        xlab="Time of Day",
        ylab="Traffic Volume",
        main="Traffic") 
# Plot together
par(mfrow=c(1,2))
# BY DayofWeek # 
boxplot(df1$NO2 ~ reorder(format(df1$dayofweek),df1$dayofweek), 
        outline = FALSE, col='lightblue3',
        xlab="Encoded Day of Week",
        ylab="Concentration ug/m3",
        main="NO2") 
boxplot(df1$Traffic ~ reorder(format(df1$dayofweek),df1$dayofweek), 
        outline = FALSE, col='darkorange2',
        xlab="Encoded Day of Week",
        ylab="Traffic Volume",
        main="Traffic") 
"""

  # c. NO2 Polar Frequency plot
# Plot wind speed and direction using PolarFreq from Openair library
# Weighted radar plot, w/ wind speed and NO2 concentration in direction of wind 
# 1. Seasonal plot
polarFreq(df1, pollutant = "NO2", type = "season", statistic = "mean", min.bin = 2)
# 2. 2020 plot
polarFreq(df1, pollutant = "NO2", type = "year", statistic = "mean", min.bin = 2)

  # d. Correlation Matrix
# plot for every variable
df2 <- df1[c('O3','NO','NO2','NOXasNO2','SO2','PM10','PM2.5','wd','ws','temp', 'Traffic')]
# plot a full cormat using the original order of data
# using pearson's correlation
rquery.cormat(df2, type="full", order = "original")

  # e. Pairplot at consistent time (12pm)
# Plotting scatter, distribution and correlation 
# collect 12pm time subset
df11 <- df1[df1$Time=='12:00:00',]
# separate into pollutants with NO2
df2 <- df11[c('NO2','NO','NOXasNO2','O3','PM10','PM2.5','SO2')]
# and meteorological & traffic with NO2
df3 <- df11[c('NO2','ws','temp', 'Traffic', 'msl', 'wd')]
# plot with line of best fit on lower scatter figure
ggpairs(df2, lower=list(continuous=wrap("smooth", size=0.5, color="cornflowerblue")))
ggpairs(df3, lower=list(continuous=wrap("smooth", size=0.5, color="mediumpurple")))


