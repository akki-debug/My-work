import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import numpy as np

class RelianceStockAnalyzer:
    """
    A comprehensive technical analysis tool for Reliance Industries stock data.
    
    Features:
    - Moving averages calculation
    - Bollinger Bands analysis
    - MACD indicator
    - RSI calculation
    - Correlation analysis
    """
    
    def __init__(self):
        self.stock_data: Optional[pd.DataFrame] = None
        self.moving_averages: Dict[str, pd.Series] = {}
        self.rsi: Optional[pd.Series] = None
        self.bollinger_bands: Dict[str, pd.Series] = {}
        self.macd: Optional[pd.Series] = None
        self.signal: Optional[pd.Series] = None
        
    def fetch_stock_data(self, start_date: str, end_date: str) -> None:
        """Fetch Reliance Industries stock data from Yahoo Finance."""
        try:
            self.stock_data = yf.download(
                'RELIANCE.NS',
                start=start_date,
                end=end_date,
                progress=False  # Disable progress bars for cleaner output
            )
            
            if self.stock_data.empty:
                raise ValueError("No data found for the specified date range")
                
            # Clean up any missing values
            self.stock_data.fillna(method='ffill', inplace=True)
            
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")

    def calculate_moving_averages(self, windows: List[int] = [50, 100, 200]) -> Dict[str, pd.Series]:
        """Calculate multiple moving averages with customizable windows."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Please fetch data first.")
            
        self.moving_averages = {
            f'MA_{window}': self.stock_data['Close'].rolling(window=window).mean()
            for window in windows
        }
        
        return self.moving_averages

    def calculate_rsi(self, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index with customizable window."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Please fetch data first.")
            
        delta = self.stock_data['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        
        roll_up = up.rolling(window).mean()
        roll_down = down.rolling(window).mean().abs()
        
        RS = roll_up / roll_down
        RSI = 100.0 - (100.0 / (1.0 + RS))
        
        self.rsi = RSI
        return RSI

    def calculate_bollinger_bands(self, 
                                window: int = 20, 
                                num_std: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands with customizable parameters."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Please fetch data first.")
            
        rolling_mean = self.stock_data['Close'].rolling(window).mean()
        rolling_std = self.stock_data['Close'].rolling(window).std()
        
        self.bollinger_bands = {
            'Middle': rolling_mean,
            'Upper': rolling_mean + (rolling_std * num_std),
            'Lower': rolling_mean - (rolling_std * num_std)
        }
        
        return self.bollinger_bands

    def calculate_macd(self, 
                      fast_window: int = 12, 
                      slow_window: int = 26, 
                      signal_window: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal Line with customizable parameters."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Please fetch data first.")
            
        exp1 = self.stock_data['Close'].ewm(span=fast_window, adjust=False).mean()
        exp2 = self.stock_data['Close'].ewm(span=slow_window, adjust=False).mean()
        
        self.macd = exp1 - exp2
        self.signal = self.macd.ewm(span=signal_window, adjust=False).mean()
        
        return self.macd, self.signal

    def analyze_correlations(self) -> Dict[str, float]:
        """Analyze correlations between indicators and stock price."""
        if self.stock_data is None or not self.moving_averages:
            raise ValueError("Stock data and moving averages must be calculated first.")
            
        correlations = {}
        
        # Calculate correlations for moving averages
        for ma_name, ma_value in self.moving_averages.items():
            valid_data = ma_value.dropna()
            if not valid_data.empty:
                correlations[f'{ma_name} Correlation'] = valid_data.corr(self.stock_data['Close'])
        
        # Calculate RSI correlation if available
        if self.rsi is not None:
            valid_rsi = self.rsi.dropna()
            if not valid_rsi.empty:
                correlations['RSI Correlation'] = valid_rsi.corr(self.stock_data['Close'])
        
        # Calculate Bollinger Bands correlations if available
        if self.bollinger_bands:
            for band_name, band_value in self.bollinger_bands.items():
                valid_band = band_value.dropna()
                if not valid_band.empty:
                    correlations[f'{band_name} Correlation'] = valid_band.corr(self.stock_data['Close'])
        
        # Calculate MACD correlation if available
        if self.macd is not None:
            valid_macd = self.macd.dropna()
            if not valid_macd.empty:
                correlations['MACD Correlation'] = valid_macd.corr(self.stock_data['Close'])
        
        return correlations

def main():
    """Main Streamlit application for Reliance Stock Technical Analysis."""
    st.title('Reliance Stock Technical Analysis')
    
    # Parameters section
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            'Start Date', 
            value=datetime(2020, 1, 1),
            min_value=datetime(2010, 1, 1)
        )
    with col2:
        end_date = st.date_input(
            'End Date', 
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Initialize analyzer
    try:
        analyzer = RelianceStockAnalyzer()
        analyzer.fetch_stock_data(start_date, end_date)
        
        # Calculate indicators with customizable parameters
        analyzer.calculate_moving_averages([50, 100, 200])
        analyzer.calculate_rsi(window=14)
        analyzer.calculate_bollinger_bands(window=20, num_std=2.0)
        analyzer.calculate_macd(fast_window=12, slow_window=26, signal_window=9)
        
        # Create visualization figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Stock Price with Moving Averages
        analyzer.stock_data['Close'].plot(ax=ax1, label='Reliance Close Price', color='black')
        for ma_name, ma_value in analyzer.moving_averages.items():
            ma_value.plot(ax=ax1, label=ma_name)
        ax1.set_title('Stock Price with Moving Averages')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Plot 2: Bollinger Bands
        analyzer.stock_data['Close'].plot(ax=ax2, label='Close Price', color='blue')
        for band_name, band_value in analyzer.bollinger_bands.items():
            band_value.plot(ax=ax2, label=band_name)
        ax2.set_title('Bollinger Bands')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # Plot 3: MACD and Signal Line
        analyzer.macd.plot(ax=ax3, label='MACD', color='blue')
        analyzer.signal.plot(ax=ax3, label='Signal Line', color='orange')
        ax3.set_title('MACD and Signal Line')
        ax3.legend(loc='upper left')
        ax3.grid(True)
        
        # Plot 4: RSI
        analyzer.rsi.plot(ax=ax4, label='RSI', color='purple')
        ax4.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
        ax4.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
        ax4.set_title('Relative Strength Index (RSI)')
        ax4.legend(loc='upper left')
        ax4.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display correlations
        st.header('Indicator Correlations with Stock Price')
        correlations = analyzer.analyze_correlations()
        correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
        correlation_df.index.name = 'Indicator'
        st.table(correlation_df.style.format('{:.3f}'))
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
