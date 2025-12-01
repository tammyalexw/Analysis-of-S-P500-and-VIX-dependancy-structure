#Load Packages 
library(readr)
library(dplyr)
library(lubridate)
library(lmtest)
library(sandwich)
library(quantreg)
library(vars)
library(rugarch)

############################################################
# 0. Load data ONCE into df and sort chronologically
############################################################

file_path = "../data/VIX_historical_data.csv"
vix = read.csv(file_path)

file_path_1 = "../data/GSPC_historical_data.csv"
gspc = read.csv(file_path_1)

gspc = gspc %>%
  mutate(Date = dmy(Date)) %>%
  arrange(Date) %>%
  select(Date, Close_GSPC)


vix  <- vix %>%
  mutate(Date = dmy(Date)) %>%
  arrange(Date) %>%
  select(Date, Close_vix)


# 1) Keep only overlapping dates
merged <- inner_join(gspc, vix, by = "Date") %>% arrange(Date)

# 2) FROM 1992 1 Jan onwards 
merged_df <- merged %>%
  mutate(Date = as_date(Date)) %>%           # safe if not already Date
  filter(Date >= ymd("1992-01-01")) %>%      # keep 1992-01-01 onward
  arrange(Date)

# Make returns 
merged_df = merged_df %>%
  mutate(
    rsp  = 100 * (log(Close_GSPC) - log(lag(Close_GSPC))),
    rvix = 100 * (log(Close_vix)  - log(lag(Close_vix)))
  ) %>%
  filter(!is.na(rsp), !is.na(rvix))

# Final df 
df_result = merged_df %>%
  select(Date, rsp, rvix)

# Quick sanity check
summary(df_result[, c("rsp","rvix")])

# Run csv
#write.csv(df_result, "../data/sp500_vix_returns.csv", row.names = FALSE)
