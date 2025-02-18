import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Dict, Tuple, List
import time

class OptionsStrategy:
    def __init__(self):
        self.transaction_cost = 0.001  # 0.1% transaction cost
        self.risk_free_rate = 0.06     # 6% risk-free rate
        
    def calculate_implied_volatility(self, S: float, K: float, T: float, 
                                   r: float, market_price: float, 
                                   option_type: str = 'call') -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        def black_scholes(S: float, K: float, T: float, r: float, 
                         sigma: float, option_type: str) -> float:
            d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            if option_type == 'call':
                return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        def objective(sigma: float) -> float:
            return black_scholes(S, K, T, r, sigma, option_type) - market_price
        
        def derivative(sigma: float) -> float:
            h = 1e-4
            return (objective(sigma + h) - objective(sigma - h))/(2*h)
        
        # Newton-Raphson iteration
        sigma = 0.2  # Initial guess
        max_iter = 100
        for _ in range(max_iter):
            if abs(objective(sigma)) < 1e-6:
                break
            if abs(derivative(sigma)) < 1e-6:
                raise ValueError("Derivative too close to zero")
            sigma = sigma - objective(sigma)/derivative(sigma)
            
        return sigma
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate 50-day, 100-day, and 200-day moving averages"""
        return {
            'MA_50': prices.rolling(window=50).mean(),
            'MA_100': prices.rolling(window=100).mean(),
            'MA_200': prices.rolling(window=200).mean()
        }
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        excess_returns = returns - self.risk_free_rate/252
        sharpe = excess_returns.mean()/returns.std()*np.sqrt(252)
        return sharpe
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int, int, int]:
        """Calculate maximum drawdown and its duration"""
        peak = equity_curve.cummax()
        drawdown = equity_curve/peak - 1
        max_dd = drawdown.min()
        max_dd_end = drawdown.idxmin()
        max_dd_start = equity_curve[:max_dd_end].idxmax()
        
        return abs(max_dd), len(equity_curve[max_dd_start:max_dd_end]), max_dd_start, max_dd_end

class StrategyBacktest:
    def __init__(self):
        self.exceptions = []
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
    def get_options_data(self, ticker: str, date: datetime) -> pd.DataFrame:
        """Get options data for Reliance with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                opt = yf.Ticker(f"{ticker}.NS")
                options = opt.options
                
                if not options:
                    raise ValueError("No options data available")
                
                df = pd.DataFrame()
                for expiry in options[:3]:
                    opt_data = opt.option_chain(expiry)
                    
                    if opt_data.calls.empty or opt_data.puts.empty:
                        continue
                    
                    calls = opt_data.calls[['strike', 'impliedVolatility', 'lastPrice']]
                    puts = opt_data.puts[['strike', 'impliedVolatility', 'lastPrice']]
                    
                    calls['expiry'] = pd.to_datetime(expiry)
                    puts['expiry'] = pd.to_datetime(expiry)
                    
                    df = pd.concat([df, calls, puts])
                
                if df.empty:
                    raise ValueError("No valid options data found")
                
                return df.sort_values('expiry')
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.exceptions.append(f"Error getting options data (attempt {attempt + 1}): {str(e)}")
                    return pd.DataFrame()
                time.sleep(self.retry_delay)
                continue
    
    def execute_strategy(self, ticker: str, start_date: datetime, end_date: datetime):
        """Execute the strategy with improved error handling"""
        try:
            stock_data = yf.download(ticker + '.NS', start=start_date, end=end_date)
            
            if stock_data.empty:
                raise ValueError("No stock data available")
            
            strategy = OptionsStrategy()
            backtest = StrategyBacktest()
            
            mas = strategy.calculate_moving_averages(stock_data['Close'])
            
            equity_curve = pd.Series(index=stock_data.index)
            positions = pd.DataFrame(index=stock_data.index)
            
            monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
            
            for date in monthly_dates:
                opt_data = backtest.get_options_data(ticker, date)
                
                if opt_data.empty:
                    continue
                    
                current_price = stock_data.loc[date, 'Close']
                otm_calls = opt_data[opt_data['strike'] > current_price]
                itm_puts = opt_data[opt_data['strike'] < current_price]
                
                if len(otm_calls) > 0 and len(itm_puts) > 0:
                    otm_strike = otm_calls.iloc[0]['strike']
                    itm_strike = itm_puts.iloc[-1]['strike']
                    
                    try:
                        iv_otm = strategy.calculate_implied_volatility(
                            current_price, otm_strike, 
                            (opt_data['expiry'].iloc[0] - date).days/252,
                            strategy.risk_free_rate, 
                            opt_data[opt_data['strike'] == otm_strike].iloc[0]['lastPrice'],
                            'call'
                        )
                        
                        iv_itm = strategy.calculate_implied_volatility(
                            current_price, itm_strike,
                            (opt_data['expiry'].iloc[0] - date).days/252,
                            strategy.risk_free_rate,
                            opt_data[opt_data['strike'] == itm_strike].iloc[0]['lastPrice'],
                            'put'
                        )
                        
                        positions.loc[date, 'OTM_Call_Strike'] = otm_strike
                        positions.loc[date, 'ITM_Put_Strike'] = itm_strike
                        positions.loc[date, 'OTM_Call_IV'] = iv_otm
                        positions.loc[date, 'ITM_Put_IV'] = iv_itm
                    except Exception as e:
                        self.exceptions.append(f"Error calculating IV at {date}: {str(e)}")
                        continue
            
            return {
                'moving_averages': mas,
                'positions': positions,
                'equity_curve': equity_curve,
                'exceptions': backtest.exceptions
            }
        except Exception as e:
            self.exceptions.append(f"Error executing strategy: {str(e)}")
            return None

def create_streamlit_app():
    """Create Streamlit app for strategy visualization"""
    st.title("Reliance Options Trading Strategy")
    
    ticker = st.text_input("Enter Reliance ticker symbol", value="RELIANCE")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    
    if st.button("Run Backtest"):
        backtest = StrategyBacktest()
        result = backtest.execute_strategy(ticker, start_date, end_date)
        
        if result:
            # Plot moving averages
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(result['moving_averages']['MA_50'], label='MA_50')
            ax.plot(result['moving_averages']['MA_100'], label='MA_100')
            ax.plot(result['moving_averages']['MA_200'], label='MA_200')
            ax.legend()
            st.pyplot(fig)
            
            # Display positions
            st.write("Strategy Positions:")
            st.table(result['positions'])
            
            # Display exceptions
            if result['exceptions']:
                st.write("Exceptions encountered:")
                for exception in result['exceptions']:
                    st.error(exception)
                    
            # Display performance metrics
            if not result['equity_curve'].empty:
                sharpe = OptionsStrategy().calculate_sharpe_ratio(result['equity_curve'])
                max_dd, dd_duration, dd_start, dd_end = OptionsStrategy().calculate_max_drawdown(result['equity_curve'])
                
                st.write("\nPerformance Metrics:")
                st.write(f"Sharpe Ratio: {sharpe:.2f}")
                st.write(f"Maximum Drawdown: {max_dd*100:.2f}%")
                st.write(f"Drawdown Duration: {dd_duration} days")
                st.write(f"Drawdown Period: {dd_start.strftime('%Y-%m-%d')} to {dd_end.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    create_streamlit_app()
