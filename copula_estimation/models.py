import numpy as np
import pandas as pd
from scipy import stats

'''
General Pipeline: 
1. Estimate parameters for each distribution using stats.fit() (returns MLE params)
2. Compute log-likelihood using stats.logpdf() using MLE params 
3. Calculate AIC and BIC
'''
# Load data
df = pd.read_csv("../data/sp500_vix_returns.csv")
data_sp = df["rsp"]
data_vix = df["rvix"]
n_sp = len(data_sp)
n_vix = len(data_vix)

''' Model fitting and comparison for rsp (S&P 500 returns) '''

# Fit Gaussian (Normal) distribution 
# Params: mu, sigma
print("Fitting Gaussian distribution...")
mu, sigma = stats.norm.fit(data_sp) # params maximise likelihood
loglik_gaussian = np.sum(stats.norm.logpdf(data_sp, loc=mu, scale=sigma)) 
k_gaussian = 2  
aic_gaussian = 2 * k_gaussian - 2 * loglik_gaussian 
bic_gaussian = k_gaussian * np.log(n_sp) - 2 * loglik_gaussian 
print(f"  mu = {mu:.6f}, sigma = {sigma:.6f}")
print(f"  Log-likelihood = {loglik_gaussian:.2f}")
print(f"  AIC = {aic_gaussian:.2f}, BIC = {bic_gaussian:.2f}\n")

# Fit Student-t distribution
# Params: mu, sigma, df 
# Note: df here refers to the df for the best empirical fit for data, not # of params in distribution. Continuous. 
print("Fitting Student-t distribution...")
df_t, mu_t, sigma_t = stats.t.fit(data_sp)
loglik_t = np.sum(stats.t.logpdf(data_sp, df=df_t, loc=mu_t, scale=sigma_t))
k_t = 3  
aic_t = 2 * k_t - 2 * loglik_t
bic_t = k_t * np.log(n_sp) - 2 * loglik_t
print(f"  df = {df_t:.6f}, mu = {mu_t:.6f}, sigma = {sigma_t:.6f}")
print(f"  Log-likelihood = {loglik_t:.2f}")
print(f"  AIC = {aic_t:.2f}, BIC = {bic_t:.2f}\n")

# Fit Skewed Student-t distribution
# Params: mu, sigma, df, skewness 
print("Fitting Skewed Student-t distribution...")
skew_params = stats.jf_skew_t.fit(data_sp)
a_skewt, b_skewt, mu_skewt, sigma_skewt = skew_params # a & b collectively control df & skewness. no direct mapping of params
loglik_skewt = np.sum(stats.jf_skew_t.logpdf(data_sp, a=a_skewt, b=b_skewt, 
                                              loc=mu_skewt, scale=sigma_skewt))
k_skewt = 4 
aic_skewt = 2 * k_skewt - 2 * loglik_skewt
bic_skewt = k_skewt * np.log(n_sp) - 2 * loglik_skewt
print(f"  a = {a_skewt:.6f}, b = {b_skewt:.6f}")
print(f"  mu = {mu_skewt:.6f}, sigma = {sigma_skewt:.6f}")
print(f"  Log-likelihood = {loglik_skewt:.2f}")
print(f"  AIC = {aic_skewt:.2f}, BIC = {bic_skewt:.2f}\n")

# Fit Laplace distribution
# Params: mu, b (scale)
print("Fitting Laplace distribution...")
mu_laplace, scale_laplace = stats.laplace.fit(data_sp) 
loglik_laplace = np.sum(stats.laplace.logpdf(data_sp, loc=mu_laplace, scale=scale_laplace))
k_laplace = 2 
aic_laplace = 2 * k_laplace - 2 * loglik_laplace
bic_laplace = k_laplace * np.log(n_sp) - 2 * loglik_laplace
print(f"  loc = {mu_laplace:.6f}, scale = {scale_laplace:.6f}")
print(f"  Log-likelihood = {loglik_laplace:.2f}")
print(f"  AIC = {aic_laplace:.2f}, BIC = {bic_laplace:.2f}\n")

# Create summary table
results = pd.DataFrame({
    'Distribution': ['Gaussian', 'Student-t', 'Skewed-t', 'Laplace'],
    'Parameters': [k_gaussian, k_t, k_skewt, k_laplace],
    'Log-Likelihood': [loglik_gaussian, loglik_t, loglik_skewt, loglik_laplace],
    'AIC': [aic_gaussian, aic_t, aic_skewt, aic_laplace],
    'BIC': [bic_gaussian, bic_t, bic_skewt, bic_laplace]
})

print("\n" + "="*70)
print("Summary Table: Model Comparison for S&P 500 Returns")
print("="*70)
print(results.to_string(index=False))
print("="*70)
print("Best model by AIC:", results.loc[results['AIC'].idxmin(), 'Distribution'])
print("Best model by BIC:", results.loc[results['BIC'].idxmin(), 'Distribution'])


''' Model fitting and comparison for rvix (VIX returns) '''

# Fit Gaussian (Normal) distribution for VIX
print("\nFitting Gaussian distribution...")
mu_vix, sigma_vix = stats.norm.fit(data_vix)
loglik_gaussian_vix = np.sum(stats.norm.logpdf(data_vix, loc=mu_vix, scale=sigma_vix))
k_gaussian_vix = 2
aic_gaussian_vix = 2 * k_gaussian_vix - 2 * loglik_gaussian_vix
bic_gaussian_vix = k_gaussian_vix * np.log(n_vix) - 2 * loglik_gaussian_vix
print(f"  mu = {mu_vix:.6f}, sigma = {sigma_vix:.6f}")
print(f"  Log-likelihood = {loglik_gaussian_vix:.2f}")
print(f"  AIC = {aic_gaussian_vix:.2f}, BIC = {bic_gaussian_vix:.2f}")

# Fit Student-t distribution for VIX
print("\nFitting Student-t distribution...")
df_t_vix, mu_t_vix, sigma_t_vix = stats.t.fit(data_vix)
loglik_t_vix = np.sum(stats.t.logpdf(data_vix, df=df_t_vix, loc=mu_t_vix, scale=sigma_t_vix))
k_t_vix = 3
aic_t_vix = 2 * k_t_vix - 2 * loglik_t_vix
bic_t_vix = k_t_vix * np.log(n_vix) - 2 * loglik_t_vix
print(f"  df = {df_t_vix:.6f}, mu = {mu_t_vix:.6f}, sigma = {sigma_t_vix:.6f}")
print(f"  Log-likelihood = {loglik_t_vix:.2f}")
print(f"  AIC = {aic_t_vix:.2f}, BIC = {bic_t_vix:.2f}")

# Fit Skewed Student-t distribution for VIX
print("\nFitting Skewed Student-t distribution...")
skew_params_vix = stats.jf_skew_t.fit(data_vix)
a_skewt_vix, b_skewt_vix, mu_skewt_vix, sigma_skewt_vix = skew_params_vix
loglik_skewt_vix = np.sum(stats.jf_skew_t.logpdf(data_vix, a=a_skewt_vix, b=b_skewt_vix,
                                                   loc=mu_skewt_vix, scale=sigma_skewt_vix))
k_skewt_vix = 4
aic_skewt_vix = 2 * k_skewt_vix - 2 * loglik_skewt_vix
bic_skewt_vix = k_skewt_vix * np.log(n_vix) - 2 * loglik_skewt_vix
print(f"  a = {a_skewt_vix:.6f}, b = {b_skewt_vix:.6f}")
print(f"  mu = {mu_skewt_vix:.6f}, sigma = {sigma_skewt_vix:.6f}")
print(f"  Log-likelihood = {loglik_skewt_vix:.2f}")
print(f"  AIC = {aic_skewt_vix:.2f}, BIC = {bic_skewt_vix:.2f}")

# Fit Laplace distribution for VIX
print("\nFitting Laplace distribution...")
loc_laplace_vix, scale_laplace_vix = stats.laplace.fit(data_vix)
loglik_laplace_vix = np.sum(stats.laplace.logpdf(data_vix, loc=loc_laplace_vix, scale=scale_laplace_vix))
k_laplace_vix = 2
aic_laplace_vix = 2 * k_laplace_vix - 2 * loglik_laplace_vix
bic_laplace_vix = k_laplace_vix * np.log(n_vix) - 2 * loglik_laplace_vix
print(f"  loc = {loc_laplace_vix:.6f}, scale = {scale_laplace_vix:.6f}")
print(f"  Log-likelihood = {loglik_laplace_vix:.2f}")
print(f"  AIC = {aic_laplace_vix:.2f}, BIC = {bic_laplace_vix:.2f}")

# Create summary table for VIX
results_vix = pd.DataFrame({
    'Distribution': ['Gaussian', 'Student-t', 'Skewed-t', 'Laplace'],
    'Parameters': [k_gaussian_vix, k_t_vix, k_skewt_vix, k_laplace_vix],
    'Log-Likelihood': [loglik_gaussian_vix, loglik_t_vix, loglik_skewt_vix, loglik_laplace_vix],
    'AIC': [aic_gaussian_vix, aic_t_vix, aic_skewt_vix, aic_laplace_vix],
    'BIC': [bic_gaussian_vix, bic_t_vix, bic_skewt_vix, bic_laplace_vix]
})

print("\n" + "="*70)
print("Summary Table: Model Comparison for VIX Returns")
print("="*70)
print(results_vix.to_string(index=False))
print("="*70)
print("Best model by AIC:", results_vix.loc[results_vix['AIC'].idxmin(), 'Distribution'])
print("Best model by BIC:", results_vix.loc[results_vix['BIC'].idxmin(), 'Distribution'])


