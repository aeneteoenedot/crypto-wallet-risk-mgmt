# 🪙 Crypto Market Risk Toolkit

https://github.com/user-attachments/assets/ed6975d0-ba1a-42be-aac9-1422562f7bd9

The **Crypto Risk Toolkit** is a Python-based script for managing cryptocurrency price data for Bitcoin, Ethereum and Solana, it also analyzes some basic market risk metrics, and portfolio risk. It includes tools to calculate:

- Exponentially Weighted Moving Average (EWMA) volatility and covariance
- Parametric and historical Value at Risk (VaR)
- Stress test impacts
- Portfolio expected return and volatility

Price data is sourced directly from Binance and stored locally using **SQLite** for simplicity and portability.

A lightweight **Django web app** (`myapp`) is included to visualize the analysis results and serve a basic front end for exploring portfolio risk metrics.

Ideal for crypto traders, analysts, and developers interested in quantifying and monitoring the risk of portfolios comprised of one of those 3 crypto-currencies.
The code can be further modified to include any currency available in `ccxt` package

## 📊 Key Features

- **Data Collection & Storage**
  - Fetches historical OHLCV data from Binance using `ccxt`.
  - Stores price and returns data in an SQLite database.
  - Supports incremental updates and bulk inserts via DataFrames.

- **Risk Analysis**
  - EWMA-based volatility, covariance, and correlation matrices.
  - Portfolio-level return and volatility metrics.
  - Parametric and historical VaR estimations.
  - Stress testing with user-defined scenarios.

- **Web-Based Visualization**
  - Django web app for displaying portfolio statistics and risk metrics.
  - Interactive charts and dashboards powered by Plotly and Pandas.

## 🚀 Getting Started

These instructions will get you up and running with the Crypto Risk Toolkit on your local machine.

### 📦 Prerequisites

Make sure you have the following installed:

- Python 3.8+
- pip
- virtualenv (recommended)
- SQLite (default DB)
- Node.js & npm (for Django static assets, optional)

### 🔧 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/crypto-risk-toolkit.git
cd crypto-risk-toolkit
```



