import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
from scipy.optimize import newton

# Fetch Reliance Stock Data
def fetch_stock_data(ticker="RELIANCE.NS", period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

# Fetch Reliance Options Chain
def fetch_options_data(ticker="RELIANCE.NS"):
    stock = yf.Ticker(ticker)
    expiries = stock.options  # Get available expirations
    latest_expiry = expiries[0]  # Nearest expiry
    options_chain = stock.option_chain(latest_expiry)
    return options_chain.calls, options_chain.puts

# Black-Scholes Model for Option Pricing
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

# Goal Seek to Get Implied Volatility
def implied_volatility(S, K, T, r, market_price, option_type="call"):
    func = lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price
    try:
        return newton(func, 0.2)  # Initial guess at 20%
    except RuntimeError:
        return np.nan  # If Newton fails

# Strategy: Buy OTM Call, Short ITM Put
def options_trading_strategy():
    stock_data = fetch_stock_data()
    calls, puts = fetch_options_data()
    latest_price = stock_data["Close"].iloc[-1]

    # Select OTM Call and ITM Put
    otm_call = calls[calls["strike"] > latest_price].iloc[0]
    itm_put = puts[puts["strike"] < latest_price].iloc[-1]

    # Option parameters
    K_otm = otm_call["strike"]
    K_itm = itm_put["strike"]
    T = 30 / 365  # 1 month expiry
    r = 0.05  # Risk-free rate

    # Compute Implied Volatility
    iv_otm = implied_volatility(latest_price, K_otm, T, r, otm_call["lastPrice"], "call")
    iv_itm = implied_volatility(latest_price, K_itm, T, r, itm_put["lastPrice"], "put")

    if np.isnan(iv_otm) or np.isnan(iv_itm):
        return "Failed to compute IVs."

    # Compute Black-Scholes Price
    price_otm = black_scholes(latest_price, K_otm, T, r, iv_otm, "call")
    price_itm = black_scholes(latest_price, K_itm, T, r, iv_itm, "put")

    # Capital Simulation
    capital = 100000  # Starting capital
    transaction_cost = 0.001  # 0.1% per trade
    capital -= price_otm + transaction_cost * price_otm  # Buying OTM Call
    capital += price_itm - transaction_cost * price_itm  # Shorting ITM Put

    return {
        "Stock Price": latest_price,
        "OTM Call Strike": K_otm,
        "ITM Put Strike": K_itm,
        "OTM IV": iv_otm,
        "ITM IV": iv_itm,
        "OTM Call Price": price_otm,
        "ITM Put Price": price_itm,
        "Remaining Capital": capital
    }

# Backtesting Strategy
def backtest():
    stock_data = fetch_stock_data(period="2y")
    stock_data["MA_50"] = stock_data["Close"].rolling(50).mean()
    stock_data["MA_100"] = stock_data["Close"].rolling(100).mean()
    stock_data["MA_200"] = stock_data["Close"].rolling(200).mean()

    returns = stock_data["Close"].pct_change()
    cumulative_returns = (1 + returns).cumprod()
    
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    max_drawdown = np.max(1 - cumulative_returns / cumulative_returns.cummax())

    return stock_data, sharpe_ratio, max_drawdown

# Streamlit Dashboard
st.title("Reliance Options Trading Strategy")
st.write("Buying **OTM Calls** and Shorting **ITM Puts** at the start of the month.")

# Fetch Data & Strategy Execution
strategy_results = options_trading_strategy()
st.write("### Strategy Execution Results")
st.write(strategy_results)

# Backtesting & Risk Metrics
stock_data, sharpe_ratio, max_drawdown = backtest()

# Moving Averages Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(stock_data.index, stock_data["Close"], label="Stock Price", color="blue")
ax.plot(stock_data.index, stock_data["MA_50"], label="50-Day MA", color="red", linestyle="dashed")
ax.plot(stock_data.index, stock_data["MA_100"], label="100-Day MA", color="green", linestyle="dashed")
ax.plot(stock_data.index, stock_data["MA_200"], label="200-Day MA", color="purple", linestyle="dashed")
ax.legend()
st.pyplot(fig)

# Performance Metrics
st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
st.write(f"**Maximum Drawdown:** {max_drawdown:.2%}")

