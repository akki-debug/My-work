import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Optional

class RelianceStockAnalyzer:
    def __init__(self):
        self.stock_data: Optional[pd.DataFrame] = None
        self.moving_averages: Dict[str, pd.Series] = {}
        
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
        
    def calculate_moving_averages(self) -> Dict[str, pd.Series]:
        """Calculate 50, 100, and 200 day moving averages"""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Please fetch data first.")
            
        self.moving_averages = {
            'MA_50': self.stock_data['Close'].rolling(window=50).mean(),
            'MA_100': self.stock_data['Close'].rolling(window=100).mean(),
            'MA_200': self.stock_data['Close'].rolling(window=200).mean()
        }
        return self.moving_averages

def main():
    """Main Streamlit application"""
    st.title('Reliance Stock Price with Moving Averages')
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', 
                                 value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input('End Date', 
                               value=datetime(2024, 12, 31))
    
    # Initialize and run analysis
    try:
        analyzer = RelianceStockAnalyzer()
        analyzer.fetch_stock_data(start_date, end_date)
        analyzer.calculate_moving_averages()
        
        # Plot stock price with moving averages
        st.header('Stock Price with Moving Averages')
        fig, ax = plt.subplots(figsize=(12, 6))
        analyzer.stock_data['Close'].plot(ax=ax, label='Reliance Close Price')
        for ma_name, ma_value in analyzer.moving_averages.items():
            ma_value.plot(ax=ax, label=ma_name)
        ax.set_title('Reliance Stock Price with Moving Averages')
        ax.legend(loc='upper left')
        ax.grid(True)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
