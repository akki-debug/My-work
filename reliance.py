import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
from scipy.optimize import newton
from datetime import datetime, timedelta

# Simulated Data Generation
np.random.seed(42)
dates = pd.date_range(start="2022-01-01", periods=500, freq="D")
reliance_prices = np.cumsum(np.random.randn(500) * 2 + 0.5) + 250  # Simulated stock price
option_prices = reliance_prices * 0.05 + np.random.randn(500) * 5  # Simulated options data

data = pd.DataFrame({"Date": dates, "Stock_Price": reliance_prices, "Option_Price": option_prices})
data.set_index("Date", inplace=True)

# Moving Averages
data["MA_50"] = data["Stock_Price"].rolling(50).mean()
data["MA_100"] = data["Stock_Price"].rolling(100).mean()
data["MA_200"] = data["Stock_Price"].rolling(200).mean()

# Black-Scholes-Merton Model for Implied Volatility
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

# Goal Seek to Match Market Price
def implied_volatility(S, K, T, r, market_price, option_type="call"):
    func = lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price
    try:
        return newton(func, 0.2)  # Initial guess at 20%
    except RuntimeError:
        return np.nan  # If Newton fails

# Strategy: Buy OTM options, Short ITM options at the start of the month
def strategy(data):
    monthly_positions = []
    capital = 100000  # Initial capital
    transaction_cost = 0.001  # 0.1% per trade
    returns = []

    for i in range(1, len(data)):
        if data.index[i].day == 1:  # Start of the month
            stock_price = data.iloc[i]["Stock_Price"]
            K_otm = stock_price * 1.05  # OTM Strike Price
            K_itm = stock_price * 0.95  # ITM Strike Price
            T = 30 / 365  # 1 month expiry
            r = 0.05  # Risk-free rate
            
            # Calculate Implied Volatility
            iv_otm = implied_volatility(stock_price, K_otm, T, r, data.iloc[i]["Option_Price"], "call")
            iv_itm = implied_volatility(stock_price, K_itm, T, r, data.iloc[i]["Option_Price"], "put")

            if np.isnan(iv_otm) or np.isnan(iv_itm):
                continue  # Skip if IV calculation fails

            price_otm = black_scholes(stock_price, K_otm, T, r, iv_otm, "call")
            price_itm = black_scholes(stock_price, K_itm, T, r, iv_itm, "put")

            capital -= price_otm + transaction_cost * price_otm
            capital += price_itm - transaction_cost * price_itm

            monthly_positions.append((data.index[i], price_otm, price_itm, capital))

        returns.append(capital)

    return pd.DataFrame(monthly_positions, columns=["Date", "Buy_OTM", "Short_ITM", "Capital"])

# Backtesting and Risk Metrics
def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
    daily_returns = np.diff(returns) / returns[:-1]
    excess_returns = daily_returns - (risk_free_rate / 252)
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def max_drawdown(returns):
    peak = np.maximum.accumulate(returns)
    drawdown = (peak - returns) / peak
    return np.max(drawdown)

# Backtesting
strategy_results = strategy(data)
sharpe_ratio = calculate_sharpe_ratio(strategy_results["Capital"].values)
max_dd = max_drawdown(strategy_results["Capital"].values)

# Streamlit App
st.title("Reliance Options Trading Strategy")
st.write("Buying OTM options and shorting ITM options at the start of the month.")

# Plot Moving Averages
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data["Stock_Price"], label="Stock Price", color="blue")
ax.plot(data.index, data["MA_50"], label="50-Day MA", color="red", linestyle="dashed")
ax.plot(data.index, data["MA_100"], label="100-Day MA", color="green", linestyle="dashed")
ax.plot(data.index, data["MA_200"], label="200-Day MA", color="purple", linestyle="dashed")
ax.legend()
st.pyplot(fig)

# Show Strategy Performance
st.subheader("Strategy Performance")
st.write(strategy_results)

# Plot Capital Growth
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(strategy_results["Date"], strategy_results["Capital"], label="Capital", color="black")
ax2.set_title("Capital Growth Over Time")
ax2.legend()
st.pyplot(fig2)

# Metrics
st.write(f"**Annualized Sharpe Ratio:** {sharpe_ratio:.2f}")
st.write(f"**Maximum Drawdown:** {max_dd:.2%}")
