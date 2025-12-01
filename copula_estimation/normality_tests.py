import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from scipy.stats import anderson

df = pd.read_csv("../data/sp500_vix_returns.csv")

## Tests for sp500 distribution

# Jarque bera test: Reject null hypothesis of normality if p < 0.05
stat, p = jarque_bera(df["rsp"])  
p_str = f"{p:.4f}" if p > 1e-10 else "< 1e-10"
print(f"Jarque-Bera test statistic for S&P 500 returns: {stat:.4f}, p-value: {p_str}")

# Anderson-Darling test: Reject null hypothesis of normality if A^2 > critical value at chosen significance level
res = anderson(df["rsp"], dist='norm')
crit_vals = res.critical_values    
sig_levels = res.significance_level  # array of [%]: [15, 10, 5, 2.5, 1]

print(f"SP500 A^2: {res.statistic:.4f}")
print("critical values:", crit_vals)
print("significance levels (%):", sig_levels)


## Tests for vix distribution
# Jarque bera test: Reject null hypothesis of normality if p < 0.05
stat, p = jarque_bera(df["rvix"])  
p_str = f"{p:.4f}" if p > 1e-10 else "< 1e-10"
print(f"Jarque-Bera test statistic for VIX returns: {stat:.4f}, p-value: {p_str}")

# Anderson-Darling test: Reject null hypothesis of normality if A^2 > critical value at chosen significance level
res = anderson(df["rvix"], dist='norm')
crit_vals = res.critical_values   
sig_levels = res.significance_level  # array of [%]: [15, 10, 5, 2.5, 1]

print(f"VIX A^2: {res.statistic:.4f}")
print("critical values:", crit_vals)
print("significance levels (%):", sig_levels)
