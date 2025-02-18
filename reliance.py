import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Dict, List, Tuple

class RelianceOptionsStrategy:
    def __init__(self):
        self.stock_data = None
        self.moving_averages = {}
        self.option_prices = {}
        self.positions = []
        
    def fetch_stock_data(self, start_date: str, end_date: str) -> None:
        """Fetch Reliance Industries stock data"""
        try:
            self.stock_data = yf.download('RELIANCE.NS', 
                                        start=start_date, 
                                        end=end_date)
            if self.stock_data.empty:
                raise ValueError("No data found for the specified date range")
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")
        
    def calculate_moving_averages(self) -> Dict[str, float]:
        """Calculate 50, 100, and 200 day moving averages"""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Please fetch data first.")
            
        self.moving_averages = {
            'MA_50': self.stock_data['Close'].rolling(window=50).mean(),
            'MA_100': self.stock_data['Close'].rolling(window=100).mean(),
            'MA_200': self.stock_data['Close'].rolling(window=200).mean()
        }
        return self.moving_averages
        
    def bsm_option_price(self, 
                        S: float, 
                        K: float, 
                        T: float, 
                        r: float, 
                        sigma: float, 
                        option_type: str) -> float:
        """Calculate European option price using Black-Scholes-Merton model"""
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
    def implied_volatility(self, 
                          market_price: float, 
                          S: float, 
                          K: float, 
                          T: float, 
                          r: float, 
                          option_type: str) -> float:
        """Calculate implied volatility using binary search"""
        tol = 1e-5
        sigma_min, sigma_max = 0.001, 2.0
        
        while sigma_max - sigma_min > tol:
            sigma_mid = (sigma_min + sigma_max)/2
            theoretical_price = self.bsm_option_price(
                S, K, T, r, sigma_mid, option_type)
            
            # Convert to float for comparison
            theoretical_price = float(theoretical_price)
            market_price = float(market_price)
            
            if abs(theoretical_price - market_price) < tol:
                return sigma_mid
                
            elif theoretical_price < market_price:
                sigma_min = sigma_mid
            else:
                sigma_max = sigma_mid
                
        return (sigma_min + sigma_max)/2

def execute_strategy(strategy: RelianceOptionsStrategy,
                   transaction_cost: float = 0.001,
                   risk_free_rate: float = 0.05) -> Dict:
    """Execute the options trading strategy"""
    positions = []
    equity_curve = []
    current_equity = 1000000  # Initial capital in INR
    
    for date in pd.date_range(start=strategy.stock_data.index[0], 
                             end=strategy.stock_data.index[-1],
                             freq='MS'):  # Monthly frequency
        
        current_price = strategy.stock_data.loc[date, 'Close']
        ma_values = {k: v.loc[date] for k, v in strategy.moving_averages.items()}
        
        atm_strike = round(current_price / 100) * 100
        otm_call_strike = atm_strike + 100
        itm_put_strike = atm_strike - 100
        
        try:
            iv_call = strategy.implied_volatility(
                market_price=100,  # Assuming standardized contract size
                S=current_price,
                K=otm_call_strike,
                T=30/365,  # One month expiry
                r=risk_free_rate,
                option_type='call')
            
            iv_put = strategy.implied_volatility(
                market_price=100,
                S=current_price,
                K=itm_put_strike,
                T=30/365,
                r=risk_free_rate,
                option_type='put')
            
            call_price = strategy.bsm_option_price(
                current_price, otm_call_strike, 30/365, 
                risk_free_rate, iv_call, 'call')
            
            put_price = strategy.bsm_option_price(
                current_price, itm_put_strike, 30/365, 
                risk_free_rate, iv_put, 'put')
            
            trade_size = int(0.01 * current_equity / current_price)
            
            positions.append({
                'date': date,
                'long_otm_call': {'strike': otm_call_strike, 
                                'price': call_price,
                                'size': trade_size},
                'short_itm_put': {'strike': itm_put_strike,
                                'price': put_price,
                                'size': trade_size},
                'transaction_cost': transaction_cost * trade_size * current_price
            })
            
            current_equity -= (call_price - put_price) * trade_size * 100 + \
                             positions[-1]['transaction_cost']
            equity_curve.append(current_equity)
            
        except Exception as e:
            st.error(f"Error processing date {date}: {str(e)}")
            continue
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    annualized_return = np.mean(returns) * 252
    annualized_volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    peak_equity = np.maximum.accumulate(equity_curve)
    drawdown = (peak_equity - equity_curve) / peak_equity
    max_drawdown = np.max(drawdown)
    
    return {
        'positions': positions,
        'equity_curve': equity_curve,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'annual_return': annualized_return,
        'annual_volatility': annualized_volatility
    }

def main():
    """Main Streamlit application"""
    st.title('Reliance Options Trading Strategy Analyzer')
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', 
                                 value=datetime(2020, 1, 1))
        transaction_cost = st.slider('Transaction Cost (%)', 
                                   min_value=0.0, 
                                   max_value=1.0, 
                                   value=0.1, 
                                   step=0.01)
    with col2:
        end_date = st.date_input('End Date', 
                               value=datetime(2024, 12, 31))
        risk_free_rate = st.slider('Risk Free Rate (%)', 
                                 min_value=0.0, 
                                 max_value=10.0, 
                                 value=5.0, 
                                 step=0.1)
    
    # Initialize and run strategy
    try:
        strategy = RelianceOptionsStrategy()
        strategy.fetch_stock_data(start_date, end_date)
        strategy.calculate_moving_averages()
        results = execute_strategy(strategy, 
                                 transaction_cost=transaction_cost/100,
                                 risk_free_rate=risk_free_rate/100)
        
        # Display results
        st.header('Strategy Performance')
        metrics = {
            'Annual Return': f"{results['annual_return']*100:.2f}%",
            'Annual Volatility': f"{results['annual_volatility']*100:.2f}%",
            'Sharpe Ratio': f"{results['sharpe_ratio']:.2f}",
            'Maximum Drawdown': f"{results['max_drawdown']*100:.2f}%"
        }
        for metric, value in metrics.items():
            st.write(f"{metric}: {value}")
        
        # Plot stock price with moving averages
        st.header('Stock Price with Moving Averages')
        fig, ax = plt.subplots(figsize=(12, 6))
        strategy.stock_data['Close'].plot(ax=ax, label='Reliance Close Price')
        for ma_name, ma_value in strategy.moving_averages.items():
            ma_value.plot(ax=ax, label=ma_name)
        ax.set_title('Reliance Stock Price with Moving Averages')
        ax.legend(loc='upper left')
        ax.grid(True)
        st.pyplot(fig)
        
        # Plot equity curve
        st.header('Strategy Equity Curve')
        fig, ax = plt.subplots(figsize=(12, 6))
        pd.Series(results['equity_curve']).plot(ax=ax)
        ax.set_title('Strategy Equity Curve')
        ax.grid(True)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error executing strategy: {str(e)}")

if __name__ == "__main__":
    main()
