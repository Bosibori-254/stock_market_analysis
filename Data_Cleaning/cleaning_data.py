"""Configuration settings for market analysis"""

# Market symbols to analyze
from market_analyzer import MarketAnalyzer
from utils import setup_logging, ensure_output_directories
from datetime import datetime
from config import SYMBOLS
from visualization import Visualizer
from technical_analysis import TechnicalAnalyzer
from data_fetcher import DataFetcher
from config import CHART_FIGSIZE, CHART_DPI, OUTPUT_DIR, CHART_DIR
import os
from typing import Dict
import matplotlib.pyplot as plt
from config import MA_SHORT_PERIOD, MA_LONG_PERIOD
from typing import Dict, Tuple
from config import SYMBOLS, ANALYSIS_PERIOD, MIN_DATA_POINTS
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf
SYMBOLS = {
    'SPY': 'S&P 500 ETF',
    'QQQ': 'NASDAQ-100 ETF',
    'DIA': 'Dow Jones ETF',
    'IWM': 'Russell 2000 ETF',
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.'
}

# Analysis parameters
ANALYSIS_PERIOD = '5y'
MA_SHORT_PERIOD = 50
MA_LONG_PERIOD = 200
MIN_DATA_POINTS = 200

# Chart settings
CHART_FIGSIZE = (14, 10)
CHART_DPI = 300

# File paths
OUTPUT_DIR = 'output'
CHART_DIR = 'charts'

# ===================================

# data_fetcher.py
"""Data fetching and preprocessing module"""


logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles market data fetching and basic preprocessing"""

    def __init__(self, symbols: Optional[Dict[str, str]] = None):
        self.symbols = symbols or SYMBOLS

    def fetch_symbol_data(self, symbol: str, period: str = ANALYSIS_PERIOD) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol"""
        try:
            logger.info(f"Fetching data for {symbol}")
            data = yf.download(symbol, period=period, auto_adjust=True)

            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            if len(data) < MIN_DATA_POINTS:
                logger.warning(
                    f"Insufficient data for {symbol}: {len(data)} records")
                return None

            logger.info(
                f"Successfully fetched {len(data)} records for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured symbols"""
        results = {}

        for symbol, name in self.symbols.items():
            data = self.fetch_symbol_data(symbol)
            if data is not None:
                results[symbol] = data

        logger.info(f"Successfully fetched data for {len(results)} symbols")
        return results

# ===================================


# technical_analysis.py
"""Technical analysis calculations"""


logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Handles technical analysis calculations"""

    def __init__(self, ma_short: int = MA_SHORT_PERIOD, ma_long: int = MA_LONG_PERIOD):
        self.ma_short = ma_short
        self.ma_long = ma_long

    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        data = data.copy()
        data['MA50'] = data['Close'].rolling(window=self.ma_short).mean()
        data['MA200'] = data['Close'].rolling(window=self.ma_long).mean()
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MA crossover"""
        data = data.copy()

        # Generate signals
        signal_condition = data['MA50'] > data['MA200']
        data['Signal'] = signal_condition.astype(int)

        # Handle NaN values
        data.loc[data['MA50'].isna() | data['MA200'].isna(), 'Signal'] = 0

        # Position changes (buy/sell points)
        data['Position'] = data['Signal'].diff()

        return data

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily and strategy returns"""
        data = data.copy()

        # Daily returns
        data['Daily_Return'] = data['Close'].pct_change()

        # Strategy returns (signal lag by 1 day)
        data['Strategy_Return'] = data['Signal'].shift(
            1) * data['Daily_Return']

        return data

    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            # Total return (buy & hold)
            total_return = float(
                data['Close'].iloc[-1] / data['Close'].iloc[0] - 1)

            # Strategy returns
            strategy_returns = data['Strategy_Return'].dropna()

            if len(strategy_returns) > 0:
                strategy_cum_return = float(
                    (1 + strategy_returns).cumprod().iloc[-1] - 1)

                # Sharpe ratio
                if strategy_returns.std() > 0:
                    sharpe_ratio = float(strategy_returns.mean(
                    ) / strategy_returns.std() * np.sqrt(252))
                else:
                    sharpe_ratio = 0.0
            else:
                strategy_cum_return = 0.0
                sharpe_ratio = 0.0

            return {
                'total_return': total_return,
                'strategy_return': strategy_cum_return,
                'sharpe_ratio': sharpe_ratio
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_return': 0.0,
                'strategy_return': 0.0,
                'sharpe_ratio': 0.0
            }

# ===================================


# visualization.py
"""Chart generation and visualization"""


logger = logging.getLogger(__name__)


class Visualizer:
    """Handles chart generation and visualization"""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        self.chart_dir = os.path.join(output_dir, CHART_DIR)
        self._ensure_directories()

    def _ensure_directories(self):
        """Create output directories if they don't exist"""
        os.makedirs(self.chart_dir, exist_ok=True)

    def plot_symbol_analysis(self, symbol: str, data: pd.DataFrame, name: str) -> str:
        """Generate analysis chart for a single symbol"""
        try:
            plt.figure(figsize=CHART_FIGSIZE)

            # Price and moving averages subplot
            plt.subplot(2, 1, 1)
            self._plot_price_and_mas(data)
            self._plot_signals(data)
            plt.title(f'{symbol} - Price and Moving Average Crossover Strategy',
                      fontsize=14, fontweight='bold')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Cumulative returns subplot
            plt.subplot(2, 1, 2)
            self._plot_cumulative_returns(data)
            plt.title('Cumulative Returns Comparison',
                      fontsize=14, fontweight='bold')
            plt.ylabel('Growth of $1')
            plt.xlabel('Date')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save chart
            filename = f'{symbol}_analysis.png'
            filepath = os.path.join(self.chart_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=CHART_DPI, bbox_inches='tight')
            plt.close()

            logger.info(f"Chart saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error creating chart for {symbol}: {e}")
            return ""

    def _plot_price_and_mas(self, data: pd.DataFrame):
        """Plot price and moving averages"""
        plt.plot(data.index, data['Close'],
                 label='Price', linewidth=2, color='black')
        plt.plot(data.index, data['MA50'],
                 label='50-Day MA', linewidth=1.5, color='blue')
        plt.plot(data.index, data['MA200'],
                 label='200-Day MA', linewidth=1.5, color='red')

    def _plot_signals(self, data: pd.DataFrame):
        """Plot buy/sell signals"""
        buy_signals = data[data['Position'] == 1]
        sell_signals = data[data['Position'] == -1]

        if not buy_signals.empty:
            plt.scatter(buy_signals.index, buy_signals['Close'],
                        color='green', marker='^', s=100, label='Buy Signal', alpha=0.7)
        if not sell_signals.empty:
            plt.scatter(sell_signals.index, sell_signals['Close'],
                        color='red', marker='v', s=100, label='Sell Signal', alpha=0.7)

    def _plot_cumulative_returns(self, data: pd.DataFrame):
        """Plot cumulative returns comparison"""
        cumulative_returns = (1 + data['Daily_Return']).cumprod()
        strategy_returns = (1 + data['Strategy_Return'].fillna(0)).cumprod()

        plt.plot(data.index, cumulative_returns,
                 label='Buy & Hold', linewidth=2, color='blue')
        plt.plot(data.index, strategy_returns,
                 label='MA Crossover Strategy', linewidth=2, color='green')

# ===================================


# market_analyzer.py
"""Main market analyzer class"""


logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Main class orchestrating market analysis"""

    def __init__(self, symbols: Optional[Dict[str, str]] = None):
        self.symbols = symbols or SYMBOLS
        self.data_fetcher = DataFetcher(self.symbols)
        self.technical_analyzer = TechnicalAnalyzer()
        self.visualizer = Visualizer()

        # Storage for analysis results
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.analysis_results: Dict[str, Dict] = {}

    def fetch_data(self) -> bool:
        """Fetch market data for all symbols"""
        logger.info("Starting data fetch process")
        self.raw_data = self.data_fetcher.fetch_all_data()

        if not self.raw_data:
            logger.error("No data could be fetched")
            return False

        logger.info(f"Data fetched for {len(self.raw_data)} symbols")
        return True

    def run_analysis(self) -> bool:
        """Run technical analysis on fetched data"""
        if not self.raw_data:
            logger.error("No data available for analysis")
            return False

        logger.info("Running technical analysis")

        for symbol, data in self.raw_data.items():
            try:
                # Calculate technical indicators
                analyzed_data = self.technical_analyzer.calculate_moving_averages(
                    data)
                analyzed_data = self.technical_analyzer.generate_signals(
                    analyzed_data)
                analyzed_data = self.technical_analyzer.calculate_returns(
                    analyzed_data)

                # Calculate performance metrics
                metrics = self.technical_analyzer.calculate_performance_metrics(
                    analyzed_data)

                # Store results
                self.analysis_results[symbol] = {
                    'data': analyzed_data,
                    'name': self.symbols[symbol],
                    **metrics
                }

                logger.info(f"Analysis completed for {symbol}")

            except Exception as e:
                logger.error(f"Analysis failed for {symbol}: {e}")

        return len(self.analysis_results) > 0

    def generate_reports(self):
        """Generate all reports and visualizations"""
        if not self.analysis_results:
            logger.error("No analysis results available")
            return

        # Generate trading signals report
        self._generate_trading_signals()

        # Generate summary statistics
        self._generate_summary_statistics()

        # Generate charts
        self._generate_charts()

    def _generate_trading_signals(self):
        """Generate trading signals report"""
        logger.info("Generating trading signals report")

        print("\n" + "="*60)
        print("TRADING SIGNALS ANALYSIS - MOVING AVERAGE CROSSOVER STRATEGY")
        print("="*60)

        for symbol, result in self.analysis_results.items():
            data = result['data']

            # Get recent signals
            recent_data = data.tail(10)
            current_signal = int(recent_data['Signal'].iloc[-1])
            previous_signal = int(recent_data['Signal'].iloc[-2])

            # Current market state
            current_price = float(data['Close'].iloc[-1])
            ma50 = float(data['MA50'].iloc[-1])
            ma200 = float(data['MA200'].iloc[-1])

            print(f"\n{symbol} ({result['name']}):")
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  50-Day MA: ${ma50:.2f}")
            print(f"  200-Day MA: ${ma200:.2f}")
            print(
                f"  Trend: {'BULLISH' if current_signal == 1 else 'BEARISH'}")

            if current_signal != previous_signal:
                action = "BUY" if current_signal == 1 else "SELL"
                print(f"  ⚡ NEW SIGNAL: {action}")

            print(f"  Buy & Hold Return: {result['total_return']*100:+.2f}%")
            print(f"  Strategy Return: {result['strategy_return']*100:+.2f}%")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")

    def _generate_summary_statistics(self):
        """Generate summary statistics table"""
        logger.info("Generating summary statistics")

        summary_data = []
        for symbol, result in self.analysis_results.items():
            summary_data.append({
                'Symbol': symbol,
                'Buy&Hold (%)': result['total_return'] * 100,
                'Strategy (%)': result['strategy_return'] * 100,
                'Sharpe Ratio': result['sharpe_ratio'],
                'Outperformance': (result['strategy_return'] - result['total_return']) * 100
            })

        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Sharpe Ratio', ascending=False)

        print("\n" + "="*60)
        print("SUMMARY STATISTICS (5-Year Performance)")
        print("="*60)
        print(df_summary.to_string(index=False,
              float_format=lambda x: f'{x:.2f}'))

        # Best performer
        if not df_summary.empty:
            best_strategy = df_summary.iloc[0]
            print(f"\n🏆 Best Performing Strategy: {best_strategy['Symbol']}")
            print(f"   Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")
            print(f"   Strategy Return: {best_strategy['Strategy (%)']:.2f}%")

    def _generate_charts(self):
        """Generate charts for all symbols"""
        logger.info("Generating charts")

        for symbol, result in self.analysis_results.items():
            chart_path = self.visualizer.plot_symbol_analysis(
                symbol, result['data'], result['name']
            )
            if chart_path:
                print(f"📊 Chart saved: {chart_path}")

    def get_market_outlook(self) -> Dict[str, int]:
        """Get current market outlook summary"""
        bullish_count = sum(
            1 for result in self.analysis_results.values()
            if result['data']['Signal'].iloc[-1] == 1
        )

        return {
            'bullish_assets': bullish_count,
            'total_assets': len(self.analysis_results),
            'bullish_percentage': (bullish_count / len(self.analysis_results)) * 100 if self.analysis_results else 0
        }

# ===================================


# utils.py
"""Utility functions"""


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(
                f'logs/market_analysis_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )


def ensure_output_directories():
    """Ensure all output directories exist"""
    from config import OUTPUT_DIR, CHART_DIR

    dirs_to_create = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, CHART_DIR),
        'logs'
    ]

    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)

# ===================================


# main.py
"""Main application entry point"""


def main():
    """Main application function"""
    # Setup
    setup_logging('INFO')
    ensure_output_directories()

    logger = logging.getLogger(__name__)

    logger.info("Starting Stock Market Analysis")
    print("Starting Stock Market Analysis...")
    print("="*60)
    print("Strategy: Moving Average Crossover (50-day vs 200-day)")
    print("="*60)

    try:
        # Initialize analyzer
        analyzer = MarketAnalyzer()

        # Fetch data
        if not analyzer.fetch_data():
            logger.error("Failed to fetch market data")
            print("❌ Failed to fetch market data. Please check your internet connection.")
            return

        # Run analysis
        if not analyzer.run_analysis():
            logger.error("Failed to run analysis")
            print("❌ Failed to run analysis on fetched data.")
            return

        # Generate reports
        analyzer.generate_reports()

        # Show market outlook
        outlook = analyzer.get_market_outlook()
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print("📈 Charts saved in output/charts/")
        print("💡 Trading signals generated based on MA crossover")
        print("📊 Performance statistics calculated")
        print("="*60)

        print(f"\nCURRENT MARKET OUTLOOK:")
        print(
            f"📊 {outlook['bullish_assets']}/{outlook['total_assets']} assets in BULLISH trend")
        print(f"📈 {outlook['bullish_percentage']:.1f}% of portfolio bullish")

        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    main()
