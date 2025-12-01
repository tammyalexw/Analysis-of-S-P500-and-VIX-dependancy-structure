import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def plot_distributions(df):
    
    rsp = df["rsp"].to_numpy()
    rvix = df["rvix"].to_numpy()
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Distribution Analysis: S&P 500 and VIX Returns", fontsize=14, fontweight='bold')
    
    # SP500 Histogram
    axes[0, 0].hist(rsp, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel("S&P 500 Returns")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("S&P 500 Returns - Histogram")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add normal distribution overlay
    mu_rsp, std_rsp = rsp.mean(), rsp.std()
    rsp_range = np.linspace(rsp.min(), rsp.max(), 100)
    axes[0, 0].plot(rsp_range, stats.norm.pdf(rsp_range, mu_rsp, std_rsp), 
                    'r-', linewidth=2, label='Normal Distribution')
    axes[0, 0].legend()
    
    # VIX Histogram
    axes[0, 1].hist(rvix, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_xlabel("VIX Returns")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("VIX Returns - Histogram")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add normal distribution overlay
    mu_rvix, std_rvix = rvix.mean(), rvix.std()
    rvix_range = np.linspace(rvix.min(), rvix.max(), 100)
    axes[0, 1].plot(rvix_range, stats.norm.pdf(rvix_range, mu_rvix, std_rvix), 
                    'r-', linewidth=2, label='Normal Distribution')
    axes[0, 1].legend()
    
    # SP500 Q-Q Plot (Sample vs Theoretical Quantiles)
    stats.probplot(rsp, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("S&P 500 Returns - Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)
    
    # VIX Q-Q Plot (Sample vs Theoretical Quantiles)
    stats.probplot(rvix, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("VIX Returns - Q-Q Plot")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # uncomment to save plots
    #plt.savefig("../data/distribution_plots.png", dpi=300, bbox_inches='tight')
    #print("\nPlots saved to ../data/distribution_plots.png")
    plt.show()

def plot_sp500_vix(sp500_csv, vix_csv):
    """
    Plot S&P 500 and VIX closing prices (1992-01-02 onward)
    in two stacked subplots within a single figure.
    """

    # ---- S&P 500 ----
    sp = pd.read_csv(sp500_csv, usecols=[0, 1], skiprows=2)
    sp.columns = ["Date", "Close"]
    sp["Date"] = pd.to_datetime(sp["Date"])
    sp["Close"] = pd.to_numeric(sp["Close"], errors="coerce")
    sp = sp.dropna(subset=["Close"])
    sp = sp[sp["Date"] >= pd.Timestamp("1992-01-02")].sort_values("Date")

    # ---- VIX ----
    vix = pd.read_csv(vix_csv, usecols=[0, 1], skiprows=2)
    vix.columns = ["Date", "Close"]
    vix["Date"] = pd.to_datetime(vix["Date"])
    vix["Close"] = pd.to_numeric(vix["Close"], errors="coerce")
    vix = vix.dropna(subset=["Close"])
    vix = vix[vix["Date"] >= pd.Timestamp("1992-01-02")].sort_values("Date")

    print(f"S&P obs: {len(sp)}, VIX obs: {len(vix)}")  # quick sanity check

    # ---- Plot both in one figure ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(sp["Date"], sp["Close"])
    axes[0].set_title("S&P 500 Closing Price")
    axes[0].set_ylabel("Price")

    axes[1].plot(vix["Date"], vix["Close"])
    axes[1].set_title("VIX Closing Price")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Price")

    plt.tight_layout()
    plt.show()

def plot_sp500_vix_returns(csv_path):
    """
    Plot S&P 500 and VIX returns against time with:
    (1) reduced visual density using alpha + thin lines
    (2) horizontal zero reference line
    """

    df = pd.read_csv(csv_path)
    df.columns = ["Date", "rsp", "rvix"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # ---- S&P 500 returns ----
    axes[0].plot(df["Date"], df["rsp"],
                 color="blue",
                 linewidth=0.6,
                 alpha=0.6)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("S&P 500 Daily Returns")
    axes[0].set_ylabel("Returns")

    # ---- VIX returns ----
    axes[1].plot(df["Date"], df["rvix"],
                 color="blue",
                 linewidth=0.6,
                 alpha=0.6)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("VIX Daily Returns")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Return")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # uncomment for returns against time plots in section 3 
    #plot_sp500_vix_returns("../data/sp500_vix_returns.csv")

    # uncomment for price against time plots in section 3
    #plot_sp500_vix(
    #    "../data/GSPC_historical_data.csv",
    #    "../data/VIX_historical_data.csv")

    # uncomment for distribution plots in section 4 
    #df = pd.read_csv("../data/sp500_vix_returns.csv")
    #plot_distributions(df)
