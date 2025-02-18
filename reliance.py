import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from scipy.stats import norm
import plotly.graph_objects as go
from typing import Dict, List, Tuple

class RelianceOptionStrategy:
    def __init__(self):
        self.stock_data = None
        self.moving_averages = {}
        self.option_prices = {}
        self.positions = []
        self.transactions = []
        
    def fetch_stock_data(self, symbol: str = 'RELIANCE.NS', period: str = '2y') -> None:
        """Fetch historical stock data from Yahoo Finance"""
        self.stock_data = yf.download(symbol, period=period)
        
    def calculate_moving_averages(self) -> None:
        """Calculate moving averages for different periods"""
        windows = [50, 100, 200]
        for window in windows:
            self.moving_averages[f'ma_{window}'] = (
                self.stock_data['Close'].rolling(window=window).mean()
            )
            
    def calculate_implied_volatility(self, 
                                   S: float,
                                   K: float,
                                   T: float,
                                   r: float,
                                   option_price: float,
                                   option_type: str = 'call') -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        def black_scholes(S: float, K: float, T: float, r: float, sigma: float, 
                         option_type: str = 'call') -> float:
            d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                
        def objective(sigma: float) -> float:
            return black_scholes(S, K, T, r, sigma, option_type) - option_price
            
        # Initial guess for volatility
        sigma = 0.2
        tolerance = 1e-5
        
        while True:
            objective_value = objective(sigma)
            derivative = (objective(sigma + tolerance) - objective_value)/tolerance
            
            sigma_new = sigma - objective_value/derivative
            if abs(sigma_new - sigma) < tolerance:
                break
                
            sigma = sigma_new
            
        return sigma
        
    def generate_trades(self, 
                       strike_range: Tuple[float, float] = (0.95, 1.05),
                       position_size: int = 100) -> List[Dict]:
        """Generate trades based on strategy rules"""
        trades = []
        current_date = datetime.now()
        
        # Get current stock price
        current_price = self.stock_data['Close'].iloc[-1]
        
        # Generate OTM and ITM strikes
        otm_strike = current_price * strike_range[0]
        itm_strike = current_price * strike_range[1]
        
        trade = {
            'date': current_date,
            'long_strike': otm_strike,
            'short_strike': itm_strike,
            'position_size': position_size,
            'strategy_type': 'vertical_spread'
        }
        
        trades.append(trade)
        return trades
        
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.04) -> float:
        """Calculate annualized Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio
        
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve.cummax()
        drawdown = equity_curve / peak - 1
        max_drawdown = drawdown.min()
        return abs(max_drawdown)
        
    def plot_strategy(self):
        """Plot strategy performance and indicators"""
        fig = go.Figure()
        
        # Plot stock price and moving averages
        fig.add_trace(go.Scatter(x=self.stock_data.index,
                                y=self.stock_data['Close'],
                                name='Stock Price'))
        
        for ma_name, ma_values in self.moving_averages.items():
            fig.add_trace(go.Scatter(x=self.stock_data.index,
                                    y=ma_values,
                                    name=ma_name))
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    # Initialize strategy
    strategy = RelianceOptionStrategy()
    
    # Fetch data
    strategy.fetch_stock_data()
    strategy.calculate_moving_averages()
    
    # Generate trades
    trades = strategy.generate_trades()
    
    # Calculate metrics
    returns = pd.Series([0.01, -0.005, 0.015])  # Example returns
    sharpe_ratio = strategy.calculate_sharpe_ratio(returns)
    max_drawdown = strategy.calculate_max_drawdown(pd.Series([100, 95, 110]))
    
    # Display results
    st.title("Reliance Options Trading Strategy")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    st.write(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    # Plot strategy
    strategy.plot_strategy()

if __name__ == "__main__":
    main()
