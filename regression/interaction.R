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
#linear model
linear_model <-lm(rsp~rvix, data = df)
summary(linear_model)
coeftest(linear_model, vcov = NeweyWest(linear_model, lag = lags_hac_robust, prewhite = FALSE))

#create dummy for sign of rvix
df_dummy = df %>% mutate(dummy = ifelse(rvix > 0, 1, 0)) %>%
  mutate(rvix_dummy = rvix * dummy)

model <- lm(rsp ~ rvix + rvix_dummy, data = df_dummy)
summary(model)
#robust SE
coeftest(model, vcov = NeweyWest(model, lag = lags_hac_robust, prewhite = FALSE))

# beta2 measures the effect of whether the sign of vix has an extra effect on rsp
#beta 2 is insignificant which means that the maginitude of effect vix has on rsp does not depend on sign


#create another interaction rvix * indicator
df_interaction = df %>% mutate(rsp_lag1 = lag(rsp,1)) %>% mutate(rsp_dummy = ifelse(rsp_lag1 > 0, 1, 0))%>% drop_na()
model1 <- lm(rsp ~ rvix + rsp_dummy, data = df_interaction)
summary(model1)
#robust SE
coeftest(model1, vcov = NeweyWest(model1, lag = lags_hac_robust, prewhite = FALSE))


#quantiles of vix returns
vix_95th <- quantile(df$rvix, 0.95)
vix_5th <- quantile(df$rvix, 0.05)
df_quantile = df %>% mutate(is_upper = ifelse(rvix >= vix_95th, 1, 0), is_lower = ifelse(rvix <= vix_5th, 1, 0)) %>%
  mutate(is_upper = rvix * is_upper, is_lower = rvix * is_lower)
model_quantile <- lm(rsp ~ rvix + is_upper + is_lower, data = df_quantile)
summary(model_quantile)
coeftest(model_quantile, vcov = NeweyWest(model_quantile, lag = lags_hac_robust, prewhite = FALSE))
