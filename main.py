
"""
This module provides tools for managing cryptocurrency price data, calculating risk metrics, and performing portfolio risk analysis.

Classes:
    CryptoData:
        - Manages cryptocurrency price data using an SQLite database.
        - Initializes and manages database connections and tables.
        - Supports inserting single data points or entire DataFrames.
        - Fetches data with calculated log returns.
        - Provides methods to truncate or drop tables and close the connection.

Functions:
    ewma_volatility(data, lambda_=0.94):
        Computes the Exponentially Weighted Moving Average (EWMA) volatility for each asset in the data.
    ewma_covariance_matrix(data, lambda_=0.94):
        Computes the EWMA covariance matrix for asset log returns.
    ewma_correlation_matrix(data, lambda_=0.94):
        Computes the EWMA correlation matrix for asset log returns.
    portfolio_volatility(weights, cov_matrix):
        Calculates the portfolio volatility given asset weights and a covariance matrix.
    parametric_var(weights, returns, cov_matrix, alpha=0.05):
        Calculates the parametric (Gaussian) Value at Risk (VaR) for a portfolio.
    stress_test_portfolio(weights, returns, shocks={'BTC/USDT': -0.2, 'ETH/USDT': -0.1, 'SOL/USDT': -0.05}):
        Applies stress scenarios to the portfolio and computes the resulting portfolio returns.
    historical_var(weights, returns, alpha=0.05):
        Calculates the historical Value at Risk (VaR) for a portfolio.
    portfolio_return(weights, returns):
        Computes the mean portfolio return given asset weights and returns data.
    exchange_fetch(symbols, time_range):
        Fetches historical OHLCV data for given symbols from Binance and stores it in the database.

Usage:
    Run the module as a script to fetch and store recent price data for BTC/USDT, ETH/USDT, and SOL/USDT.
"""

import sqlite3
import ccxt
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import norm

class CryptoData:
    """
    A class for managing cryptocurrency price data using an SQLite database.
    Attributes:
        db_path (str): Path to the SQLite database file.
        conn (sqlite3.Connection): SQLite database connection object.
        cursor (sqlite3.Cursor): Cursor object for executing SQL commands.
    Methods:
        __init__(db_path='crypto_data.db', tbl_name='crypto_data'):
            Initializes the database connection and creates the table if it does not exist.
        insert_data(ticker, date, close_price, tbl_name='crypto_data'):
            Inserts a single row of data into the specified table.
        insert_dataframe(dataframe, tbl_name='crypto_data'):
            Inserts data from a pandas DataFrame into the specified table, replacing existing data.
        fetch_data(tbl_name='crypto_data'):
            Fetches all data from the specified table, including calculated log returns for each ticker.
        truncate_data(tbl_name='crypto_data'):
            Deletes all rows from the specified table.
        drop_table(tbl_name='crypto_data'):
            Drops the specified table from the database.
        close():
            Closes the database connection.
    """
    def __init__(self, db_path='crypto_data.db', tbl_name='crypto_data'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {tbl_name} (
                ticker TEXT,
                date TEXT,
                close_price REAL
            )
        ''')
        self.conn.commit()
    
    def insert_data(self, ticker, date, close_price, tbl_name='crypto_data'):
        self.cursor.execute(f'''
            INSERT INTO {tbl_name} (ticker, date, close_price)
            VALUES (?, ?, ?)''', (ticker, date, close_price))
        self.conn.commit()

    def insert_dataframe(self, dataframe, tbl_name='crypto_data'):
        dataframe.to_sql(tbl_name, self.conn, if_exists='replace', index=False)
        
    def fetch_data(self, tbl_name='crypto_data'):
        self.cursor.execute(f"""SELECT
                                ticker,
                                date,
                                close_price,
                                CASE
                                    WHEN LAG(close_price) OVER (PARTITION BY ticker ORDER BY date) IS NULL THEN 0
                                    ELSE LOG(close_price) - LOG(LAG(close_price) OVER (PARTITION BY ticker ORDER BY date))
                                END AS log_returns
                            FROM {tbl_name}
                            ORDER BY ticker, date;""")
        data = self.cursor.fetchall()
        return data
    
    def truncate_data(self, tbl_name='crypto_data'):
        self.cursor.execute(f'DELETE FROM {tbl_name}')
        self.conn.commit()

    def drop_table(self, tbl_name='crypto_data'):
        self.cursor.execute(f'DROP TABLE IF EXISTS {tbl_name}')
        self.conn.commit()

    def close(self):
        self.conn.close()

def ewma_volatility(data, lambda_=0.94):
    """
    Calculates the exponentially weighted moving average (EWMA) volatility for each ticker in the input data.

    Parameters:
        data (pd.DataFrame or list): Input data containing columns 'ticker', 'date', 'close_price', and 'log_returns'.
                                     If a list is provided, it will be converted to a DataFrame with these columns.
        lambda_ (float, optional): Smoothing parameter for EWMA, between 0 and 1. Default is 0.94.

    Returns:
        dict: A dictionary mapping each ticker to its latest EWMA volatility value. If no returns are available for a ticker, returns np.nan.
    """
    # Ensure data is a DataFrame
    if isinstance(data, list):
        data = pd.DataFrame(data, columns=['ticker', 'date', 'close_price', 'log_returns'])
    ewma_volatility = {}
    for ticker, group in data.groupby('ticker'):
        returns = group['log_returns'].values
        ewma_var = 0
        ewma_vars = []
        for r in returns:
            ewma_var = lambda_ * ewma_var + (1 - lambda_) * r**2
            ewma_vars.append(np.sqrt(ewma_var))
        ewma_volatility[ticker] = ewma_vars[-1] if ewma_vars else np.nan
    return ewma_volatility

def ewma_covariance_matrix(data, lambda_=0.94):
    """
    Calculates the Exponentially Weighted Moving Average (EWMA) covariance matrix of asset log returns.
    Parameters:
        data (pd.DataFrame or list): Input data containing at least the columns 'ticker', 'date', and 'log_returns'.
                                     If a list is provided, it is converted to a DataFrame with columns
                                     ['ticker', 'date', 'close_price', 'log_returns'].
        lambda_ (float, optional): Smoothing factor for EWMA, between 0 and 1. Default is 0.94.
    Returns:
        pd.DataFrame: EWMA covariance matrix with assets as both index and columns.
                      Returns an empty DataFrame if input data is empty or missing required columns.
    Notes:
        - The function pivots the input data to create a returns matrix with dates as rows and tickers as columns.
        - Only rows with non-missing returns for all assets are used.
        - The EWMA covariance is recursively updated for each time step.
    """
    # Ensure data is a DataFrame
    if isinstance(data, list):
        data = pd.DataFrame(data, columns=['ticker', 'date', 'close_price', 'log_returns'])
    # Check if DataFrame is empty or missing 'log_returns'
    if data.empty or 'log_returns' not in data.columns:
        return pd.DataFrame()
    # Pivot to get returns matrix: rows=dates, columns=tickers
    pivot = data.pivot_table(index='date', columns='ticker', values='log_returns')
    aligned_returns = pivot.dropna()

    if aligned_returns.empty:
        return pd.DataFrame()
    returns = aligned_returns.to_numpy()
    n_assets = returns.shape[1]
    ewma_cov = np.zeros((n_assets, n_assets))
    for t in range(returns.shape[0]):
        r = returns[t].reshape(-1, 1)
        ewma_cov = lambda_ * ewma_cov + (1 - lambda_) * (r @ r.T)
    return pd.DataFrame(ewma_cov, index=aligned_returns.columns, columns=aligned_returns.columns)

def ewma_correlation_matrix(data, lambda_=0.94):
    """Compute the Exponentially Weighted Moving Average (EWMA) correlation matrix from log returns.

    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of assets, where each column represents an asset.
        lambda_ (float, optional): Decay factor for EWMA, between 0 and 1. Default is 0.94.

    Returns:
        pd.DataFrame: EWMA correlation matrix with the same index and columns as the input data."""

    cov_matrix = ewma_covariance_matrix(data, lambda_)
    if cov_matrix.empty:
        return pd.DataFrame()
    stddev = np.sqrt(np.diag(cov_matrix))
    # Avoid division by zero
    stddev[stddev == 0] = np.nan
    corr_matrix = cov_matrix / np.outer(stddev, stddev)
    return pd.DataFrame(corr_matrix, index=cov_matrix.index, columns=cov_matrix.columns)

def portfolio_volatility(weights, cov_matrix):
    """
    Calculates the volatility (standard deviation) of a portfolio given asset weights and a covariance matrix.

    Parameters:
        weights (array-like): A list or array of portfolio weights for each asset. The length must match the number of assets.
        cov_matrix (numpy.ndarray): A square covariance matrix representing the covariances between asset returns.

    Returns:
        float: The portfolio volatility (standard deviation).

    Raises:
        ValueError: If the covariance matrix is not square.
        ValueError: If the length of weights does not match the number of assets in the covariance matrix.

    Example:
        >>> weights = [0.5, 0.3, 0.2]
        >>> cov_matrix = np.array([[0.1, 0.02, 0.04],
        ...                        [0.02, 0.08, 0.03],
        ...                        [0.04, 0.03, 0.09]])
        >>> portfolio_volatility(weights, cov_matrix)
        0.3162
    """
    weights = np.array(weights)
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square (same number of assets for rows and columns).")
    if len(weights) != cov_matrix.shape[0]:
        raise ValueError("Weights length must match the number of assets in the covariance matrix (rows and columns).")
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance)

def parametric_var(weights, returns, cov_matrix, alpha=0.05):
    """
    Calculates the parametric Value at Risk (VaR) for a portfolio using the variance-covariance method.
    Parameters:
        weights (array-like): Portfolio asset weights.
        returns (array-like): Expected returns for each asset.
        cov_matrix (array-like): Covariance matrix of asset returns.
        alpha (float, optional): Significance level for VaR calculation (default is 0.05).
    Returns:
        float: The parametric VaR of the portfolio at the specified significance level.
    Notes:
        - Assumes returns are normally distributed.
        - Requires the functions `portfolio_return` and `portfolio_volatility` to be defined.
    """

    port_mean = portfolio_return(weights, returns)
    port_vol = portfolio_volatility(weights, cov_matrix)
    z = norm.ppf(alpha)
    var = -(port_mean + z * port_vol)
    return var

def stress_test_portfolio(weights, returns, shocks={'BTC/USDT': -0.2, 'ETH/USDT': -0.1, 'SOL/USDT': -0.05}):
    """
    Performs a stress test on a cryptocurrency portfolio by applying specified shocks to the mean log returns of selected assets.
    Args:
        weights (array-like): Portfolio weights for each asset, ordered to match the tickers in `returns`.
        returns (pd.DataFrame): DataFrame containing at least 'ticker' and 'log_returns' columns.
        shocks (dict, optional): Dictionary mapping tickers to shock values (as additive changes to mean log returns). 
            Defaults to {'BTC/USDT': -0.2, 'ETH/USDT': -0.1, 'SOL/USDT': -0.05}.
    Returns:
        dict: A dictionary with:
            - 'shocked_portfolio_return': The portfolio return after applying the shocks.
            - 'difference': The difference between the shocked and original portfolio returns.
    Raises:
        KeyError: If any ticker in `weights` does not match the tickers in `returns`.
        ValueError: If the length of `weights` does not match the number of unique tickers in `returns`.
    Example:
        >>> stress_test_portfolio([0.5, 0.3, 0.2], returns_df)
    """

    tickers = returns['ticker'].unique()
    shocked_returns = returns.groupby('ticker')['log_returns'].mean().copy()
    for i, ticker in enumerate(tickers):
        if ticker in shocks:
            shocked_returns.loc[ticker] += shocks[ticker]
    shocked_portfolio_return = np.dot(weights, shocked_returns.values)
    # Calculate original portfolio return
    original_returns = returns.groupby('ticker')['log_returns'].mean().values
    original_portfolio_return = np.dot(weights, original_returns)
    return {
        'shocked_portfolio_return': shocked_portfolio_return,
        'difference': shocked_portfolio_return - original_portfolio_return
    }

def historical_var(weights, returns, alpha=0.05):
    """
    Calculates the historical Value at Risk (VaR) of a portfolio using historical simulation.

    Parameters:
        weights (array-like): Portfolio weights for each asset, in the same order as the columns in the returns DataFrame.
        returns (pd.DataFrame): DataFrame containing columns 'date', 'ticker', and 'log_returns' representing asset returns.
        alpha (float, optional): Significance level for VaR calculation (default is 0.05 for 95% VaR).

    Returns:
        float: The historical VaR at the specified confidence level (as a positive number).

    Notes:
        - Assumes that 'returns' contains log returns for each asset.
        - The function aligns returns by date and ticker, dropping any dates with missing data.
        - VaR is returned as a positive value representing the potential loss.
    """
    # Pivot to get returns matrix: rows=dates, columns=tickers
    pivot = returns.pivot_table(index='date', columns='ticker', values='log_returns')
    aligned_returns = pivot.dropna()
    # Portfolio returns as weighted sum of asset returns
    port_returns = aligned_returns.values @ np.array(weights)
    var = -np.percentile(port_returns, alpha * 100)
    return var

def portfolio_return(weights, returns):
    """
    Calculates the expected portfolio return as the weighted average of mean log returns for each asset.

    Parameters:
        weights (array-like): Portfolio weights for each asset. Must match the number of unique tickers in 'returns'.
        returns (pd.DataFrame or list): Asset returns data. If a DataFrame, must contain columns ['ticker', 'date', 'close_price', 'log_returns'].
                                        If a list, it will be converted to a DataFrame with these columns.

    Returns:
        float: The expected portfolio return as the weighted sum of mean log returns per asset.

    Raises:
        ValueError: If the number of weights does not match the number of unique assets (tickers) in the returns data.
    """
    weights = np.array(weights, dtype=float)
    # Ensure 'returns' is a DataFrame
    if isinstance(returns, list):
        returns = pd.DataFrame(returns, columns=['ticker', 'date', 'close_price', 'log_returns'])
    # Compute mean log return per asset
    mean_returns = returns.groupby('ticker')['log_returns'].mean().values
    if len(weights) != len(mean_returns):
        raise ValueError("Weights length must match the number of unique assets (tickers) in the returns DataFrame.")
    return np.dot(weights, mean_returns)

def exchange_fetch(symbols, time_range):
    """
    Fetches historical OHLCV data for given cryptocurrency symbols from Binance and stores the closing prices in a database.

    Args:
        symbols (list of str): List of cryptocurrency trading pairs (e.g., ['BTC/USDT', 'ETH/USDT']).
        time_range (int): Number of days of historical data to fetch for each symbol.

    Side Effects:
        - Truncates existing data in the CryptoData table.
        - Inserts new closing price data for each symbol and date.

    Dependencies:
        - Requires the `ccxt` library for exchange access.
        - Assumes existence of a `CryptoData` class with `truncate_data` and `insert_data` methods.
        - Uses `datetime` for timestamp conversion.
    """
    tbl = CryptoData()
    tbl.truncate_data()
    exchange = ccxt.binance()
    for symbol in symbols:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=time_range)
        for entry in ohlcv:
            timestamp = entry[0]
            date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
            tbl.insert_data(symbol, date, entry[4])

if __name__ == '__main__':    
    exchange_fetch(
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        time_range = 21)

        