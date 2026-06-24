"""
Alternative Data Alpha Pipeline
Ingests simulated satellite/web-traffic/sentiment data, engineers features,
and backtests a signal against price returns.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, roc_curve)
from typing import Dict, List, Tuple, Optional


# Alternative Data Simulators 

def simulate_satellite_data(
    tickers: List[str],
    dates: pd.DatetimeIndex,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulates weekly satellite imagery counts (e.g. parking lot car counts,
    shipping container fills) for retailers/energy companies.
    Higher count → business activity above normal.
    """


    rng = np.random.default_rng(seed)
    rows = []
    # Weekly cadence; counts are noisy but correlated with future returns
    for ticker in tickers:
        base_level = rng.uniform(100, 500)
        trend = rng.uniform(-0.1, 0.2)  # long-run trend per year

        for i, date in enumerate(dates):
            # Signal: positive trend means business is growing
            signal = base_level * (1 + trend * i / 252)
            noise = rng.normal(0, base_level * 0.15)
            weekly_count = max(signal + noise, 0)
            rows.append({"date": date, "ticker": ticker, "satellite_count": weekly_count})


    df = pd.DataFrame(rows).set_index(["date", "ticker"]).unstack("ticker")
    df.columns = df.columns.get_level_values(1)
    return df





def simulate_web_traffic(
    tickers: List[str],
    dates: pd.DatetimeIndex,
    seed: int = 43,
) -> pd.DataFrame:
    """Simulates weekly web/app traffic data (page views, daily active users).
    Useful for e-commerce, SaaS companies.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for ticker in tickers:
        base = rng.uniform(1_000_000, 50_000_000)
        seasonal = rng.uniform(0.8, 1.2)
        for i, date in enumerate(dates):
            week_of_year = date.isocalendar().week
            seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * week_of_year / 52) * seasonal
            trend = 1 + rng.uniform(-0.02, 0.08) * i / 252
            traffic = base * seasonal_factor * trend * rng.lognormal(0, 0.1)
            rows.append({"date": date, "ticker": ticker, "web_traffic": traffic})


    df = pd.DataFrame(rows).set_index(["date", "ticker"]).unstack("ticker")
    df.columns = df.columns.get_level_values(1)
    return df




def simulate_social_sentiment(
    tickers: List[str],
    dates: pd.DatetimeIndex,
    seed: int = 44,
) -> pd.DataFrame:
    """
    Simulates daily social media sentiment scores (e.g. from Twitter/Reddit NLP).
    Score in [-1, +1]. Includes momentum and mean-reversion in sentiment.
    """

    rng = np.random.default_rng(seed)
    data = {}
    for ticker in tickers:
        sentiment = np.zeros(len(dates))
        s = 0.0
        for i in range(len(dates)):
            s = 0.7 * s + rng.normal(0, 0.15)  # AR(1) sentiment
            s = np.clip(s, -1, 1)
            sentiment[i] = s
        data[ticker] = sentiment
    return pd.DataFrame(data, index=dates)


def simulate_job_postings(
    tickers: List[str],
    dates: pd.DatetimeIndex,
    seed: int = 45,
) -> pd.DataFrame:
    """Simulates monthly job posting counts (from LinkedIn/Indeed scraping).
    Rising job postings → expansion → bullish.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for ticker in tickers:
        base = rng.integers(100, 2000)
        postings = [base]
        for _ in range(len(dates) - 1):
            change = rng.normal(0, base * 0.05)
            postings.append(max(postings[-1] + change, 0))
        data[ticker] = postings
    return pd.DataFrame(data, index=dates)


#  Feature Engineering 
def fetch_returns(tickers: List[str], period: str = "4y") -> pd.DataFrame:
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(0)
    return close[[t for t in tickers if t in close.columns]].pct_change()



def build_feature_matrix(
    tickers: List[str],
    returns: pd.DataFrame,
    satellite: pd.DataFrame,
    web_traffic: pd.DataFrame,
    sentiment: pd.DataFrame,
    job_postings: pd.DataFrame,
    forward_days: int = 21,
) -> pd.DataFrame:
    """
    Build a panel of alt-data features for each ticker × date.
    Target: forward return sign (up/down) over next forward_days.
    """

    all_rows = []

    for ticker in tickers:
        if ticker not in returns.columns:
            continue

        ret = returns[ticker].dropna()
        dates = ret.index

        for i in range(60, len(dates) - forward_days):
            date = dates[i]

            # Forward return label
            fwd_ret = returns[ticker].iloc[i+1:i+forward_days+1].sum()
            label = int(fwd_ret > 0)

            # Price momentum features
            ret_1w  = returns[ticker].iloc[i-5:i].sum()

            ret_1m  = returns[ticker].iloc[i-21:i].sum()

            ret_3m  = returns[ticker].iloc[i-63:i].sum()

            vol_1m  = returns[ticker].iloc[i-21:i].std() * np.sqrt(252)

            # Satellite: level change
            sat = satellite.get(ticker)
            if sat is not None:
                sat_loc = sat.index.get_indexer([date], method="nearest")[0]
                sat_chg_1m = (sat.iloc[sat_loc] / sat.iloc[max(0, sat_loc-21)] - 1) if sat_loc > 21 else 0
                sat_zscore = (sat.iloc[sat_loc] - sat.iloc[max(0, sat_loc-63):sat_loc].mean()) / \
                             (sat.iloc[max(0, sat_loc-63):sat_loc].std() + 1e-8)
            else:
                sat_chg_1m = sat_zscore = 0

            # Web traffic: YoY change (or MoM if short)
            web = web_traffic.get(ticker)
            if web is not None:
                web_loc = web.index.get_indexer([date], method="nearest")[0]
                web_chg = (web.iloc[web_loc] / web.iloc[max(0, web_loc-21)] - 1) if web_loc > 21 else 0
                web_zscore = (web.iloc[web_loc] - web.iloc[max(0, web_loc-63):web_loc].mean()) / \
                             (web.iloc[max(0, web_loc-63):web_loc].std() + 1e-8)
            else:
                web_chg = web_zscore = 0


            # Sentiment: level and momentum
            sent = sentiment.get(ticker)
            if sent is not None:
                sent_loc = sent.index.get_indexer([date], method="nearest")[0]
                sent_level = sent.iloc[sent_loc]
                sent_chg   = (sent.iloc[sent_loc] - sent.iloc[max(0, sent_loc-5)]) if sent_loc > 5 else 0
            else:
                sent_level = sent_chg = 0

            # Job postings: 3-month growth rate

            jobs = job_postings.get(ticker)
            if jobs is not None:
                job_loc = jobs.index.get_indexer([date], method="nearest")[0]
                job_growth = (jobs.iloc[job_loc] / jobs.iloc[max(0, job_loc-63)] - 1) if job_loc > 63 else 0
            else:
                job_growth = 0

            all_rows.append({
                "date": date,
                "ticker": ticker,
                # Price features

                "ret_1w": ret_1w, "ret_1m": ret_1m, "ret_3m": ret_3m, "vol_1m": vol_1m,
                # Alt data features
                "satellite_chg_1m": sat_chg_1m,
                "satellite_zscore": sat_zscore,
                "web_chg_1m": web_chg,
                "web_zscore": web_zscore,
                "sentiment_level": sent_level,
                "sentiment_chg": sent_chg,
                "job_growth_3m": job_growth,
                # Target
                "label": label,
                "fwd_return": fwd_ret,
            })

    return pd.DataFrame(all_rows)


#  Walk-Forward Backtest 

FEATURE_COLS = [
    "ret_1w", "ret_1m", "ret_3m", "vol_1m",
    "satellite_chg_1m", "satellite_zscore",
    "web_chg_1m", "web_zscore",
    "sentiment_level", "sentiment_chg",
    "job_growth_3m",
]


def walk_forward_backtest(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> Dict:
    """Time-series cross-validation with multiple classifiers."""
    df = df.sort_values("date").dropna(subset=FEATURE_COLS + ["label"])
    X = df[FEATURE_COLS].values
    y = df["label"].values
    fwd_rets = df["fwd_return"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    models = {
        "GBM":          GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                     learning_rate=0.05, random_state=42),
        "RF":           RandomForestClassifier(n_estimators=100, max_depth=5,
                                               random_state=42, n_jobs=-1),
        "Logistic":     LogisticRegression(C=0.1, max_iter=500, random_state=42),
    }

    results = {name: {"acc": [], "auc": [], "ic": [], "pnl": []} for name in models}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        ret_test = fwd_rets[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        for name, model in models.items():
            model.fit(X_train_s, y_train)
            pred_prob = model.predict_proba(X_test_s)[:, 1]
            pred_label = (pred_prob > 0.5).astype(int)

            acc = accuracy_score(y_test, pred_label)
            try:
                auc = roc_auc_score(y_test, pred_prob)
            except Exception:
                auc = 0.5

            # Information Coefficient: correlation of signal with returns

            ic = np.corrcoef(pred_prob, ret_test)[0, 1]

            # Strategy PnL: go long when pred > 0.5, short when < 0.5
            signal = 2 * pred_label - 1   # +1 / -1
            pnl = (signal * ret_test).mean() * 252   # annualized

            results[name]["acc"].append(acc)
            results[name]["auc"].append(auc)
            results[name]["ic"].append(ic)
            results[name]["pnl"].append(pnl)

    return results, df, models, scaler


def feature_importance_report(model, feature_cols: List[str]) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=feature_cols).sort_values(ascending=False)
    return pd.Series()



# Visualization 

def plot_alt_data_dashboard(df: pd.DataFrame, results: Dict,
                             ticker: str = None, save_path: str = None):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
    fig.suptitle("Alternative Data Alpha Pipeline", fontsize=14, fontweight="bold")

    #  Feature distributions (alt data vs price return deciles)
    ax1 = fig.add_subplot(gs[0, :2])
    sub = df.dropna(subset=FEATURE_COLS + ["fwd_return"])
    sub["return_decile"] = pd.qcut(sub["fwd_return"], 5, labels=False)
    grouped = sub.groupby("return_decile")["sentiment_level"].mean()
    ax1.bar(grouped.index, grouped.values, color="#3498db", alpha=0.8)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xlabel("Forward Return Quintile"); ax1.set_ylabel("Avg Sentiment")
    ax1.set_title("Sentiment Level by Forward Return Quintile")

    #  IC by model and fold

    ax2 = fig.add_subplot(gs[0, 2])
    for i, (name, res) in enumerate(results.items()):
        ax2.plot(res["ic"], marker="o", label=name, linewidth=1.5)
    ax2.axhline(0, color="black", lw=0.8, linestyle="--")
    ax2.set_xlabel("Fold"); ax2.set_ylabel("Information Coefficient")
    ax2.set_title("IC by Walk-Forward Fold")
    ax2.legend(fontsize=8)



    # AUC comparison
    ax3 = fig.add_subplot(gs[1, 0])
    names = list(results.keys())
    aucs = [np.mean(results[n]["auc"]) for n in names]
    ax3.bar(names, aucs, color=["#2ecc71", "#3498db", "#e74c3c"], alpha=0.85)
    ax3.axhline(0.5, color="black", lw=1, linestyle="--")
    ax3.set_ylabel("Mean AUC"); ax3.set_title("Mean AUC by Model")
    ax3.set_ylim(0, 1)



    # Annualized PnL
    ax4 = fig.add_subplot(gs[1, 1])
    pnls = [np.mean(results[n]["pnl"]) for n in names]
    colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pnls]
    ax4.bar(names, [p * 100 for p in pnls], color=colors, alpha=0.85)
    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_ylabel("Ann. Return (%)"); ax4.set_title("Annualized Strategy Return")


    #  Accuracy
    ax5 = fig.add_subplot(gs[1, 2])
    accs = [np.mean(results[n]["acc"]) * 100 for n in names]
    ax5.bar(names, accs, color=["#2ecc71", "#3498db", "#e74c3c"], alpha=0.85)
    ax5.axhline(50, color="black", lw=1, linestyle="--")
    ax5.set_ylabel("Accuracy (%)"); ax5.set_title("Mean Direction Accuracy")



    #Sentiment vs fwd return scatter
    ax6 = fig.add_subplot(gs[2, 0])
    sub2 = df.dropna(subset=["sentiment_level", "fwd_return"])
    ax6.scatter(sub2["sentiment_level"], sub2["fwd_return"] * 100,
                alpha=0.1, s=5, color="#8e44ad")
    ax6.set_xlabel("Sentiment Level"); ax6.set_ylabel("Forward Return (%)")
    ax6.set_title("Sentiment vs Forward Return")

    # Alt data signal distribution by outcome
    ax7 = fig.add_subplot(gs[2, 1])
    up_mask = df["label"] == 1
    down_mask = df["label"] == 0
    ax7.hist(df.loc[up_mask, "web_zscore"].dropna(), bins=40, alpha=0.5,
             color="#2ecc71", density=True, label="Up")
    ax7.hist(df.loc[down_mask, "web_zscore"].dropna(), bins=40, alpha=0.5,
             color="#e74c3c", density=True, label="Down")
    ax7.set_xlabel("Web Traffic Z-Score"); ax7.set_ylabel("Density")
    ax7.set_title("Web Traffic Z-Score by Return Direction")
    ax7.legend(fontsize=8)


    # Summary table
    ax8 = fig.add_subplot(gs[2, 2])
    summary_text = "Model Comparison\n" + "─"*28 + "\n"
    summary_text += f"{'Model':<12} {'AUC':>6} {'IC':>7} {'Ann.Ret':>8}\n"
    summary_text += "─"*36 + "\n"
    for name in names:
        a = np.mean(results[name]["auc"])
        ic = np.mean(results[name]["ic"])
        p = np.mean(results[name]["pnl"])
        summary_text += f"{name:<12} {a:>6.3f} {ic:>7.4f} {p*100:>7.1f}%\n"
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
             va="top", ha="left", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax8.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()


# Main 

def run(plot: bool = True):
    print("\n" + "="*60)
    print("  Alternative Data Alpha Pipeline")
    print("="*60 + "\n")

    TICKERS = ["AMZN", "WMT", "TGT", "HD", "COST", "SBUX", "MCD"]

    print(f"► Fetching price data for {TICKERS} …")
    returns = fetch_returns(TICKERS, period="4y")
    dates = returns.index
    available = [t for t in TICKERS if t in returns.columns]
    print(f"  {len(dates)} trading days, {len(available)} tickers available")

    print("\n Simulating alternative data sources …")
    satellite  = simulate_satellite_data(available, dates)
    web        = simulate_web_traffic(available, dates)
    sentiment  = simulate_social_sentiment(available, dates)
    job_posts  = simulate_job_postings(available, dates)
    print(f"  Satellite data:  {satellite.shape}")
    print(f"  Web traffic:     {web.shape}")
    print(f"  Social sentiment:{sentiment.shape}")
    print(f"  Job postings:    {job_posts.shape}")

    print("\n Engineering features …")
    df = build_feature_matrix(available, returns, satellite, web, sentiment,
                               job_posts, forward_days=21)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  Feature matrix: {df.shape[0]:,} rows, {len(FEATURE_COLS)} features")
    print(f"  Label balance: {df['label'].mean():.2%} positive (up)")

    print("\n Running 5-fold walk-forward backtest …")
    results, df, models, scaler = walk_forward_backtest(df, n_splits=5)

    print("\n Results:")
    print(f"  {'Model':<14} {'Accuracy':>10} {'AUC':>8} {'IC':>8} {'Ann.Ret':>10}")
    print("  " + "─"*52)
    for name, res in results.items():
        print(f"  {name:<14} {np.mean(res['acc']):>10.3f} "
              f"{np.mean(res['auc']):>8.3f} "
              f"{np.mean(res['ic']):>8.4f} "
              f"{np.mean(res['pnl'])*100:>9.1f}%")

    # Feature importance (GBM)
    print("\n Feature Importances (GBM):")
    fi = feature_importance_report(models["GBM"], FEATURE_COLS)
    for feat, imp in fi.items():
        print(f"  {feat:<25}: {imp:.4f}")

    if plot:
        print("\n Generating dashboard …")
        plot_alt_data_dashboard(df, results, save_path="alt_data_dashboard.png")

    return results, df


if __name__ == "__main__":
    run(plot=True)
