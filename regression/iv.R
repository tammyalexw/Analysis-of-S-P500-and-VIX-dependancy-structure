#load library
library(fredr)
library(dplyr)
library(tidyr)
library(purrr)
library(zoo) 
library(lubridate)
library(data.table)
library(hdm)
library(glmnet)
library(forecast)
library(readxl)
library(urca)
library(ggplot2)
library(AER)
library(moments)
library(sandwich)
library(lmtest)



#read in data
file_path = "../data/sp500_vix_returns.csv"
df = read.csv(file_path) %>% rename (date = Date) %>% mutate(date = as.Date(date))
nobs = df %>% nrow()
lags_hac_robust = floor(nobs ** (1/3))
#epu data for IV
epu_df = read.csv("../data/All_Daily_Policy_Data.csv") %>% 
  mutate(date = as.Date(sprintf("%04d-%02d-%02d", year, month, day))) %>% 
  select(date, daily_policy_index) %>% rename(epu = daily_policy_index)

#join the data on date and transfrom epu using log percentage change
df_epu = df %>% left_join(epu_df, by ='date') %>% mutate(epu = 100 * (log(epu) - log(dplyr::lag(epu)))) %>% drop_na()

#check if epu is stationary
ndiffs(df_epu[["epu"]], test = "adf") #stationary

#plot of epu  
ggplot(df_epu, aes(x = date, y = epu)) +
  geom_line() +
  labs(
    x = "Date",
    y = "US Daily Percentage Change in EPU index",
  )

x <- df_epu$epu

epu_stats <- c(
  mean     = mean(x, na.rm = TRUE),
  sd       = sd(x, na.rm = TRUE),
  var      = var(x, na.rm = TRUE),
  min      = min(x, na.rm = TRUE),
  max      = max(x, na.rm = TRUE),
  skewness = skewness(x, na.rm = TRUE),
  kurtosis = kurtosis(x, na.rm = TRUE)
)

epu_stats

#2sls

#first stage
fs <- lm(rvix ~ epu, data = df_epu)
summary(fs)
df_epu$rvix_hat <- fitted(fs)
#second stage
ss <- lm(rsp ~ rvix_hat, data = df_epu)
summary(ss)

#need to use robust SE
iv_model <- ivreg(
  rsp ~ rvix | epu, 
  data = df_epu
)
summary(iv_model, diagnostics = TRUE)
coeftest(iv_model, vcov = NeweyWest(iv_model, lag = lags_hac_robust, prewhite = FALSE))




A# use another IV candidate, lag of rvix
df_lag = df %>% mutate(rvix_lag1 = lag(rvix,1)) %>% drop_na()

#2sls
#first stage
fs1 <- lm(rvix ~ rvix_lag1, data = df_lag)
summary(fs1)
df_lag$rvix_hat <- fitted(fs1)

#2nd stage
ss1 <- lm(rsp ~ rvix_hat, data = df_lag)
summary(ss1)

#use lag1 of rvix as IV
iv_model1 <- ivreg(
  rsp ~ rvix | rvix_lag1,              
  data = df_lag
)
summary(iv_model, diagnostics = TRUE)
coeftest(iv_model1, vcov = NeweyWest(iv_model1, lag = lags_hac_robust, prewhite = FALSE)) #hac robust se

