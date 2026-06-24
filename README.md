# Alternative Data Alpha Pipeline

Ingests simulated alternative data (satellite imagery, web traffic, social sentiment, job postings), engineers predictive features, and backtests a direction-prediction signal using walk-forward cross-validation across 7 US consumer stocks.

---

## Overview

Traditional price/volume data is commoditized — every quant fund has it. The edge lies in non-traditional signals that lead fundamentals: satellite car counts predict retailer foot traffic before the earnings report, web traffic predicts SaaS growth, job postings signal expansion plans. This project builds the full pipeline from raw alt data ingestion to walk-forward backtest.

---

## How It Works

### 1. Universe

7 US consumer/retail stocks: AMZN, WMT, TGT, HD, COST, SBUX, MCD

### 2. Alternative Data Sources

| Source | What It Measures | Simulation Method |
|--------|-----------------|-------------------|
| **Satellite imagery** | Weekly business activity (parking lots, shipping containers) | Trend + noise around a base count |
| **Web traffic** | Weekly page views / daily active users | Seasonal + growth trend + lognormal noise |
| **Social sentiment** | Daily social media score in [−1, +1] | AR(1) process with mean reversion |
| **Job postings** | Monthly open positions (LinkedIn / Indeed proxy) | Random walk with drift |

### 3. Feature Engineering

For each (ticker, date) pair:

| Feature | Type | Description |
|---------|------|-------------|
| `ret_1w` | Price | 1-week return |
| `ret_1m` | Price | 1-month return |
| `ret_3m` | Price | 3-month return |
| `vol_1m` | Price | 1-month realized volatility |
| `satellite_chg_1m` | Alt | 1-month change in activity count |
| `satellite_zscore` | Alt | 63-day z-score of activity count |
| `web_chg_1m` | Alt | 1-month change in web traffic |
| `web_zscore` | Alt | 63-day z-score of web traffic |
| `sentiment_level` | Alt | Current sentiment score |
| `sentiment_chg` | Alt | 5-day change in sentiment |
| `job_growth_3m` | Alt | 3-month job posting growth rate |

**Target:** Binary sign of 21-day forward return (1 = up, 0 = down)

### 4. Models

| Model | Type |
|-------|------|
| **GBM** | Gradient Boosting Classifier (100 trees, depth 3) |
| **RF** | Random Forest Classifier (100 trees, depth 5) |
| **Logistic** | Logistic Regression (L2, C=0.1) |

### 5. Walk-Forward Cross-Validation

`TimeSeriesSplit` with 5 folds — strictly respects temporal ordering to avoid look-ahead bias. Each fold trains on all prior data and tests on the next period.

**Metrics reported:**
- **Accuracy** — directional prediction accuracy
- **AUC** — area under the ROC curve
- **IC (Information Coefficient)** — Pearson correlation between predicted probability and realized return
- **Annualized return** — long/short strategy return (`signal = 2 × predicted_label − 1`)

---

## Results

### Model Performance (5-fold Walk-Forward)

| Model | Accuracy | AUC | IC | Ann. Return |
|-------|----------|-----|----|-------------|
| GBM | 52.6% | 0.479 | −0.022 | — |
| RF | 56.5% | 0.489 | −0.008 | — |
| **Logistic** | 55.8% | **0.504** | −0.005 | — |

With simulated alt data (no planted predictive signal vs prices), AUC ≈ 0.5 is the expected result. In production with real proprietary data sources, ICs of 0.02–0.05 are considered strong.

### Feature Importances (GBM)

| Feature | Importance |
|---------|-----------|
| Job growth (3M) | 23.6% |
| Volatility (1M) | 22.8% |
| Return (3M) | 22.2% |
| Return (1M) | 13.0% |
| Web traffic change (1M) | 4.3% |
| Sentiment level | 3.4% |
| Satellite change (1M) | 2.4% |

Price-based features dominate (as expected without a real alt-data edge), but alt-data features (web traffic, sentiment, satellite) collectively contribute ~15% of importance.

---

## Output Plots

| File | Description |
|------|-------------|
| `alt_data_dashboard.png` | 8-panel dashboard: sentiment by return quintile, IC by fold and model, AUC comparison, strategy return, accuracy, sentiment distribution by direction, web traffic z-score analysis, model summary table |

---

## Usage

```bash
pip install numpy pandas scikit-learn matplotlib seaborn yfinance
python alt_data_pipeline.py
```

Data is fetched live via `yfinance`. Requires an internet connection.

---

## Replacing Simulated Data with Real Sources

| Data Type | Real Source |
|-----------|-------------|
| Satellite imagery | Orbital Insight, Descartes Labs, RS Metrics |
| Web traffic | SimilarWeb API, SEMrush |
| Social sentiment | Twitter API + FinBERT, StockTwits |
| Job postings | Revelio Labs, LinkedIn Talent Insights, Thinknum |

Plug real data in by replacing the `simulate_*()` functions with API calls that return a `pd.DataFrame` with the same shape `(dates, tickers)`.

---

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
seaborn
yfinance
```
