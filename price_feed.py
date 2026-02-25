"""
Live price feed integration with Yahoo Finance
Simple and reliable - 10-15 minute delayed prices (free tier)
"""
import streamlit as st
import logging
from typing import Dict, Optional, List
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceFeed:
    """Handle live price feeds from Yahoo Finance"""
    
    def __init__(self):
        self.connected = True  # Yahoo Finance is always "connected" (no connection needed)
    
    def connect(self) -> bool:
        """
        Always returns True - Yahoo Finance doesn't require connection
        
        Returns:
            True (always available)
        """
        return True
    
    def get_live_prices(self, tickers: list, contract_fetcher: Optional = None) -> Dict[str, Optional[float]]:
        """
        Get live prices for multiple tickers from Yahoo Finance
        
        Args:
            tickers: List of ticker symbols
            contract_fetcher: Not used (kept for compatibility)
        
        Returns:
            Dict of {ticker: price} (price is None if unavailable)
        """
        prices = {}
        
        # Handle ticker mapping (e.g., CRCL/ETHA/SOL might need special handling)
        ticker_mapping = {
            'CRCL/ETHA/SOL': 'CRCL',  # Map to primary ticker if needed
        }
        
        for ticker in tickers:
            try:
                # Use mapping if available, otherwise use ticker as-is
                yahoo_ticker = ticker_mapping.get(ticker, ticker)
                
                # For composite tickers, try first one
                if '/' in yahoo_ticker:
                    yahoo_ticker = yahoo_ticker.split('/')[0]
                
                # Fetch price from Yahoo Finance
                stock = yf.Ticker(yahoo_ticker)
                info = stock.info
                
                # Try to get current price
                if 'currentPrice' in info and info['currentPrice']:
                    prices[ticker] = float(info['currentPrice'])
                elif 'regularMarketPrice' in info and info['regularMarketPrice']:
                    prices[ticker] = float(info['regularMarketPrice'])
                elif 'previousClose' in info and info['previousClose']:
                    prices[ticker] = float(info['previousClose'])
                else:
                    # Fallback: try to get last close from history
                    hist = stock.history(period="1d", interval="1m")
                    if not hist.empty:
                        prices[ticker] = float(hist['Close'].iloc[-1])
                    else:
                        prices[ticker] = None
                        logger.warning(f"Could not get price for {ticker} (yahoo: {yahoo_ticker})")
                
            except Exception as e:
                logger.warning(f"Could not get price for {ticker}: {e}")
                prices[ticker] = None
        
        return prices
    
    def get_option_prices(self, contract_fetcher) -> Dict[str, Optional[float]]:
        """
        Get live prices for option contracts (not implemented for Yahoo Finance)
        
        Args:
            contract_fetcher: ContractFetcher instance (not used)
        
        Returns:
            Empty dict (options not supported via Yahoo Finance free tier)
        """
        logger.info("Option prices not available via Yahoo Finance free tier")
        return {}
    
    def get_single_price(self, ticker: str) -> Optional[float]:
        """
        Get live price for single ticker
        
        Args:
            ticker: Ticker symbol
        
        Returns:
            Current price or None if unavailable
        """
        prices = self.get_live_prices([ticker])
        return prices.get(ticker)
    
    def disconnect(self):
        """No-op for Yahoo Finance (no connection to disconnect)"""
        pass
    
    def is_connected(self) -> bool:
        """Always returns True - Yahoo Finance is always available"""
        return True


@st.cache_data(ttl=300)  # Cache for 5 minutes (Yahoo Finance has 10-15 min delay anyway)
def get_cached_prices(tickers: list) -> Dict[str, Optional[float]]:
    """
    Get prices with caching (refreshes every 5 minutes)
    
    Args:
        tickers: List of tickers to fetch
    
    Returns:
        Dict of {ticker: price}
    """
    feed = PriceFeed()
    return feed.get_live_prices(tickers)


def display_price_status(is_connected: bool = True) -> None:
    """
    Display price feed status indicator in Streamlit
    
    Args:
        is_connected: Always True for Yahoo Finance (kept for compatibility)
    """
    st.success("🟢 Yahoo Finance - Prices Available (10-15 min delay)", icon="✅")
    st.caption("Using Yahoo Finance free tier - prices updated every 10-15 minutes")
