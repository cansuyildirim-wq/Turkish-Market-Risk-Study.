# Project: Turkish Market Risk & Volatility Analysis
# Description: Analysis of historical volatility and correlation for BIST 50 stocks using Python.
# Author: Cansu Yildirim (via GitHub)
# Data Source: Yahoo Finance

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
TICKERS = ['THYAO.IS', 'GARAN.IS', 'TUPRS.IS']
START_DATE = '2025-01-01'
END_DATE = '2026-01-13'
WINDOW_SIZE = 21  # 21 Trading days (1 Month)

def get_data(tickers, start, end):
    """
    Fetches adjusted closing prices from Yahoo Finance.
    """
    print(f"Fetching data for {tickers}...")
    data = yf.download(tickers, start=start, end=end)['Close']
    return data

def calculate_volatility(data, window=21):
    """
    Calculates annualized rolling volatility.
    Formula: Rolling Std Dev of Log Returns * Sqrt(252)
    """
    log_returns = np.log(data / data.shift(1))
    # Annualizing factor: Square root of 252 trading days
    rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(252)
    return rolling_vol, log_returns

def plot_volatility(vol_data):
    """
    Visualizes the historical volatility trends.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(vol_data, linewidth=2)
    plt.title('BIST 50 - Annualized Rolling Volatility (Risk Analysis)', fontsize=14)
    plt.ylabel('Volatility (Annualized)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(vol_data.columns, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_correlation(returns):
    """
    Visualizes the correlation matrix to assess diversification benefits.
    """
    plt.figure(figsize=(10, 8))
    corr_matrix = returns.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix (Diversification Analysis)', fontsize=14)
    plt.show()
    return corr_matrix

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Data Ingestion
    prices = get_data(TICKERS, START_DATE, END_DATE)
    
    # 2. Risk Calculation
    volatility, returns = calculate_volatility(prices, WINDOW_SIZE)
    
    # 3. Visualization
    print("\n--- Generating Volatility Plot ---")
    plot_volatility(volatility)
    
    print("\n--- Generating Correlation Matrix ---")
    correlation = plot_correlation(returns)
    
    # 4. Output Summary
    print("\nAnalysis Complete.")
    print("Top Correlation Insight: Check TUPRS vs Banking Sector for hedging opportunities.")
