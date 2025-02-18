import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
from scipy.optimize import newton
from nsetools import Nse

# Initialize NSE API
nse = Nse()

# Function to Fetch Reliance Stock Data
def fetch_stock_data():
    try:
        stock_info = nse.get_quote("RELIANCE")
        stock_price = stock_info["lastPrice"]
        return stock_price
    except Exception as e:
        st.error(f"Error fetching stock price: {e}")
        return None

# Function to Fetch NSE Options Data
def fetch_options_data_nse(symbol="RELIANCE"):
    try:
        stock_info = nse.get_quote(symbol)
        expiry_dates = stock_info.get("expiryDates", [])

        if not expiry_dates:
            st.error("No options expiries found on NSE.")
            return None, None

        latest_expiry = expiry_dates[0]  # Nearest expiry
        return latest_expiry
    except Exception as e:
        st.error(f"Error fetching NSE options data: {e}")
        return None, None

# Black-Scholes Model for Option Pricing
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

# Function to Compute Implied Volatility
def implied_volatility(S, K, T, r, market_price, option_type="call"):
    func = lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price
    try:
        return newton(func, 0.2)  # Initial guess at 20%
    except RuntimeError:
        return np.nan  # If Newton-Raphson fails

# Strategy: Buy OTM Call, Short ITM Put
def options_trading_strategy():
    stock_price = fetch_stock_data()
    expiry = fetch_options_data_nse()

    if stock_price is None or expiry is None:
        return {"Error": "Failed to retrieve stock or options data."}

    # Simulating options data (since NSE API doesn't provide full options chain)
    strikes = np.arange(stock_price * 0.9, stock_price * 1.1, 10)  # Simulated strikes
    otm_call_strike = strikes[strikes > stock_price][0]  # OTM Call
    itm_put_strike = strikes[strikes < stock_price][-1]  # ITM Put

    # Option parameters
    T = 30 / 365  # 1 month expiry
    r = 0.05  # Risk-free rate

    # Simulated market prices for options
    market_price_call = 15.0  # Simulated OTM call price
    market_price_put = 20.0   # Simulated ITM put price

    # Compute Implied Volatility
    iv_otm = implied_volatility(stock_price, otm_call_strike, T, r, market_price_call, "call")
    iv_itm = implied_volatility(stock_price, itm_put_strike, T, r, market_price_put, "put")

    if np.isnan(iv_otm) or np.isnan(iv_itm):
        return {"Error": "Failed to compute Implied Volatility."}

    # Compute Black-Scholes Price
    price_otm = black_scholes(stock_price, otm_call_strike, T, r, iv_otm, "call")
    price_itm = black_scholes(stock_price, itm_put_strike, T, r, iv_itm, "put")

    # Capital Simulation
    capital = 100000  # Starting capital
    transaction_cost = 0.001  # 0.1% per trade
    capital -= price_otm + transaction_cost * price_otm  # Buying OTM Call
    capital += price_itm - transaction_cost * price_itm  # Shorting ITM Put

    return {
        "Stock Price": stock_price,
        "OTM Call Strike": otm_call_strike,
        "ITM Put Strike": itm_put_strike,
        "OTM IV": iv_otm,
        "ITM IV": iv_itm,
        "OTM Call Price": price_otm,
        "ITM Put Price": price_itm,
        "Remaining Capital": capital
    }

# Backtesting Strategy
def backtest():
    stock_prices = np.random.normal(2500, 50, 252)  # Simulating stock prices
    returns = np.diff(stock_prices) / stock_prices[:-1]
    cumulative_returns = np.cumprod(1 + returns)

    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    max_drawdown = np.max(1 - cumulative_returns / np.maximum.accumulate(cumulative_returns))

    return stock_prices, sharpe_ratio, max_drawdown

# Streamlit Dashboard
st.title("Reliance Options Trading Strategy (NSE Data)")
st.write("Buying **OTM Calls** and Shorting **ITM Puts** at the start of the month.")

# Fetch Data & Strategy Execution
strategy_results = options_trading_strategy()
st.write("### Strategy Execution Results")
st.write(strategy_results)

# Backtesting & Risk Metrics
stock_prices, sharpe_ratio, max_drawdown = backtest()

# Stock Price Simulation Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(stock_prices, label="Simulated Stock Prices", color="blue")
ax.set_title("Reliance Stock Price Simulation")
ax.set_xlabel("Days")
ax.set_ylabel("Stock Price")
ax.legend()
st.pyplot(fig)

# Performance Metrics
st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
st.write(f"**Maximum Drawdown:** {max_drawdown:.2%}")


