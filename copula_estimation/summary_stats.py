import pandas as pd
import numpy as np
from scipy import stats


def summary_stats(df):
        
    rsp = df["rsp"].to_numpy()
    rvix = df["rvix"].to_numpy()
    
    # Calculate stats using scipy.stats
    mean_sp = np.mean(rsp)
    mean_vx = np.mean(rvix)

    median_sp = np.median(rsp)
    median_vx = np.median(rvix)

    sd_sp = np.std(rsp, ddof=1)
    sd_vx = np.std(rvix, ddof=1)

    min_sp = np.min(rsp)
    min_vx = np.min(rvix)

    max_sp = np.max(rsp)
    max_vx = np.max(rvix)

    p5_sp = np.percentile(rsp, 5)
    p5_vx = np.percentile(rvix, 5)

    p95_sp = np.percentile(rsp, 95)
    p95_vx = np.percentile(rvix, 95)

    var_sp = np.var(rsp, ddof=1) 
    var_vx = np.var(rvix, ddof=1)
    
    # For skew calculation, bias = FALSE corrects calculations for bias 
    skew_sp = stats.skew(rsp, bias=False, nan_policy="omit") 
    skew_vx = stats.skew(rvix, bias=False, nan_policy="omit")
    # For kurtosis calculation, fisher = TRUE gives excess kurtosis (relative to 0 for normal distribution)
    # bias = FALSE corrects calculations for bias
    kurt_sp = stats.kurtosis(rsp, fisher=True, bias=False, nan_policy="omit") 
    kurt_vx = stats.kurtosis(rvix, fisher=True, bias=False, nan_policy="omit")
    
    pearson_sp, pearson_vix = stats.pearsonr(rsp, rvix)
    
    # Summary dataframe
    summary = pd.DataFrame({
        "rsp": {
            "mean": mean_sp,
            "median": median_sp,
            "std_dev": sd_sp,
            "min": min_sp,
            "max": max_sp,
            "p5": p5_sp,
            "p95": p95_sp,
            "variance": var_sp,
            "skewness": skew_sp,
            "kurtosis": kurt_sp,
        },
        "rvix": {
            "mean": mean_vx,
            "median": median_vx,
            "std_dev": sd_vx,
            "min": min_vx,
            "max": max_vx,
            "p5": p5_vx,
            "p95": p95_vx,
            "variance": var_vx,
            "skewness": skew_vx,
            "kurtosis": kurt_vx,
        }
    })
    
    corr = pd.DataFrame(
        {"pearson_r": [pearson_sp], "pearson_pvalue": [pearson_vix]},
        index=["sp500_ret ~ vix_ret"]
    )
    
    return summary, corr

if __name__ == "__main__":
    df = pd.read_csv("../data/sp500_vix_returns.csv")
    summary, corr = summary_stats(df)
    print("Summary statistics:")
    print(summary)
    print("\nCorrelation (Pearson):")
    print(corr)
