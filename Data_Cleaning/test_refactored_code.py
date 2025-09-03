# tests/test_data_fetcher.py
"""Tests for data fetcher module"""

from utils import setup_logging
from visualization import Visualizer
import logging
import tempfile
from market_analyzer import MarketAnalyzer
from technical_analysis import TechnicalAnalyzer
import numpy as np
from data_fetcher import DataFetcher
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataFetcher(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.test_symbols = {'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft'}
        self.fetcher = DataFetcher(self.test_symbols)

    @patch('yfinance.download')
    def test_fetch_symbol_data_success(self, mock_download):
        """Test successful data fetching"""
        # Mock successful data return
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104] * 50,  # 250 records
            'Open': [99, 100, 101, 102, 103] * 50,
            'High': [101, 102, 103, 104, 105] * 50,
            'Low': [98, 99, 100, 101, 102] * 50,
            'Volume': [1000000] * 250
        })
        mock_download.return_value = mock_data

        result = self.fetcher.fetch_symbol_data('AAPL')

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 250)
        mock_download.assert_called_once_with(
            'AAPL', period='5y', auto_adjust=True)

    @patch('yfinance.download')
    def test_fetch_symbol_data_empty(self, mock_download):
        """Test handling of empty data"""
        mock_download.return_value = pd.DataFrame()

        result = self.fetcher.fetch_symbol_data('INVALID')

        self.assertIsNone(result)

    @patch('yfinance.download')
    def test_fetch_symbol_data_insufficient(self, mock_download):
        """Test handling of insufficient data"""
        # Mock data with only 50 records (less than MIN_DATA_POINTS)
        mock_data = pd.DataFrame({
            'Close': list(range(50)),
            'Open': list(range(50)),
            'High': list(range(50)),
            'Low': list(range(50)),
            'Volume': [1000] * 50
        })
        mock_download.return_value = mock_data

        result = self.fetcher.fetch_symbol_data('TEST')

        self.assertIsNone(result)

    @patch('yfinance.download')
    def test_fetch_symbol_data_exception(self, mock_download):
        """Test exception handling"""
        mock_download.side_effect = Exception("Network error")

        result = self.fetcher.fetch_symbol_data('AAPL')

        self.assertIsNone(result)

# ===================================


# tests/test_technical_analysis.py
"""Tests for technical analysis module"""


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTechnicalAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = TechnicalAnalyzer()

        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 300)
        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        self.sample_data = pd.DataFrame({
            'Close': prices,
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Volume': np.random.randint(1000000, 5000000, 300)
        }, index=dates)

    def test_calculate_moving_averages(self):
        """Test moving average calculations"""
        result = self.analyzer.calculate_moving_averages(self.sample_data)

        # Check that MA columns exist
        self.assertIn('MA50', result.columns)
        self.assertIn('MA200', result.columns)

        # Check MA values are calculated correctly
        # MA50 starts at day 50
        self.assertTrue(result['MA50'].iloc[49:].notna().all())
        # MA200 starts at day 200
        self.assertTrue(result['MA200'].iloc[199:].notna().all())

        # First 49 values should be NaN for MA50
        self.assertTrue(result['MA50'].iloc[:49].isna().all())

    def test_generate_signals(self):
        """Test signal generation"""
        data_with_ma = self.analyzer.calculate_moving_averages(
            self.sample_data)
        result = self.analyzer.generate_signals(data_with_ma)

        # Check signal columns exist
        self.assertIn('Signal', result.columns)
        self.assertIn('Position', result.columns)

        # Check signals are binary (0 or 1)
        unique_signals = result['Signal'].dropna().unique()
        self.assertTrue(all(signal in [0, 1] for signal in unique_signals))

        # Check position changes are valid (-1, 0, 1)
        unique_positions = result['Position'].dropna().unique()
        self.assertTrue(all(pos in [-1, 0, 1] for pos in unique_positions))

    def test_calculate_returns(self):
        """Test return calculations"""
        data_with_signals = self.analyzer.generate_signals(
            self.analyzer.calculate_moving_averages(self.sample_data)
        )
        result = self.analyzer.calculate_returns(data_with_signals)

        # Check return columns exist
        self.assertIn('Daily_Return', result.columns)
        self.assertIn('Strategy_Return', result.columns)

        # Check daily returns are reasonable (-50% to +50%)
        daily_returns = result['Daily_Return'].dropna()
        self.assertTrue((daily_returns > -0.5).all())
        self.assertTrue((daily_returns < 0.5).all())

    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Prepare full analysis data
        data = self.analyzer.calculate_moving_averages(self.sample_data)
        data = self.analyzer.generate_signals(data)
        data = self.analyzer.calculate_returns(data)

        metrics = self.analyzer.calculate_performance_metrics(data)

        # Check all required metrics exist
        required_metrics = ['total_return', 'strategy_return', 'sharpe_ratio']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)

        # Check reasonable ranges
        self.assertTrue(-1.0 <= metrics['total_return']
                        <= 10.0)  # -100% to 1000%
        # Reasonable Sharpe range
        self.assertTrue(-10.0 <= metrics['sharpe_ratio'] <= 10.0)

# ===================================


# tests/test_market_analyzer.py
"""Tests for main market analyzer"""


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMarketAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.test_symbols = {'TEST': 'Test Symbol'}
        self.analyzer = MarketAnalyzer(self.test_symbols)

    @patch('data_fetcher.DataFetcher.fetch_all_data')
    def test_fetch_data_success(self, mock_fetch):
        """Test successful data fetching"""
        # Mock successful data fetch
        mock_data = pd.DataFrame({
            'Close': list(range(100, 300)),
            'Open': list(range(99, 299)),
            'High': list(range(101, 301)),
            'Low': list(range(98, 298)),
            'Volume': [1000000] * 200
        })
        mock_fetch.return_value = {'TEST': mock_data}

        result = self.analyzer.fetch_data()

        self.assertTrue(result)
        self.assertIn('TEST', self.analyzer.raw_data)
        self.assertEqual(len(self.analyzer.raw_data['TEST']), 200)

    @patch('data_fetcher.DataFetcher.fetch_all_data')
    def test_fetch_data_failure(self, mock_fetch):
        """Test handling of data fetch failure"""
        mock_fetch.return_value = {}

        result = self.analyzer.fetch_data()

        self.assertFalse(result)
        self.assertEqual(len(self.analyzer.raw_data), 0)

    def test_run_analysis_no_data(self):
        """Test analysis with no data"""
        result = self.analyzer.run_analysis()

        self.assertFalse(result)
        self.assertEqual(len(self.analyzer.analysis_results), 0)

# ===================================


# tests/integration_test.py
"""Integration tests for the complete workflow"""


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestIntegration(unittest.TestCase):

    def setUp(self):
        """Set up integration test"""
        # Use a minimal symbol set for faster testing
        self.test_symbols = {'SPY': 'S&P 500 ETF'}
        self.analyzer = MarketAnalyzer(self.test_symbols)

    def test_complete_workflow(self):
        """Test the complete analysis workflow"""
        # This is a real integration test that hits the API
        # Skip if no internet connection
        try:
            # Test data fetching
            fetch_success = self.analyzer.fetch_data()

            if not fetch_success:
                self.skipTest(
                    "Could not fetch data - possibly no internet connection")

            # Test analysis
            analysis_success = self.analyzer.run_analysis()
            self.assertTrue(analysis_success)

            # Check results structure
            self.assertGreater(len(self.analyzer.analysis_results), 0)

            for symbol, result in self.analyzer.analysis_results.items():
                # Check required keys exist
                required_keys = ['data', 'name', 'total_return',
                                 'strategy_return', 'sharpe_ratio']
                for key in required_keys:
                    self.assertIn(key, result)

                # Check data has required columns
                required_columns = ['Close', 'MA50', 'MA200', 'Signal',
                                    'Position', 'Daily_Return', 'Strategy_Return']
                for col in required_columns:
                    self.assertIn(col, result['data'].columns)

        except Exception as e:
            self.fail(f"Integration test failed: {e}")

# ===================================


# tests/test_runner.py
"""Test runner script"""


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_unit_tests():
    """Run all unit tests"""
    print("Running Unit Tests...")
    print("="*50)

    # Discover and run unit tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_integration_tests():
    """Run integration tests"""
    print("\nRunning Integration Tests...")
    print("="*50)
    print("⚠️  These tests will make real API calls and may take longer")

    from test_integration import TestIntegration

    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def main():
    """Main test runner"""
    print("Market Analyzer Test Suite")
    print("="*50)

    # Run unit tests first
    unit_success = run_unit_tests()

    if unit_success:
        print("✅ Unit tests passed!")

        # Ask user if they want to run integration tests
        response = input(
            "\nRun integration tests? (requires internet) [y/N]: ")
        if response.lower() in ['y', 'yes']:
            integration_success = run_integration_tests()
            if integration_success:
                print("✅ Integration tests passed!")
            else:
                print("❌ Integration tests failed!")
        else:
            print("⏭️  Skipping integration tests")
    else:
        print("❌ Unit tests failed! Fix issues before running integration tests.")


if __name__ == "__main__":
    main()

# ===================================

# tests/manual_test.py
"""Manual testing script for step-by-step validation"""


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_data_fetcher():
    """Test data fetcher manually"""
    print("Testing Data Fetcher...")
    print("-" * 30)

    # Test with single symbol
    fetcher = DataFetcher({'SPY': 'S&P 500 ETF'})

    try:
        data = fetcher.fetch_symbol_data('SPY')
        if data is not None:
            print(f"✅ Data fetched successfully: {len(data)} records")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            print(f"   Columns: {list(data.columns)}")
            return data
        else:
            print("❌ No data returned")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_technical_analysis(data):
    """Test technical analysis manually"""
    if data is None:
        print("⏭️  Skipping technical analysis (no data)")
        return None

    print("\nTesting Technical Analysis...")
    print("-" * 30)

    try:
        analyzer = TechnicalAnalyzer()

        # Test moving averages
        data_with_ma = analyzer.calculate_moving_averages(data)
        print(f"✅ Moving averages calculated")
        print(f"   MA50 non-null values: {data_with_ma['MA50'].count()}")
        print(f"   MA200 non-null values: {data_with_ma['MA200'].count()}")

        # Test signals
        data_with_signals = analyzer.generate_signals(data_with_ma)
        print(f"✅ Signals generated")

        signal_counts = data_with_signals['Signal'].value_counts()
        print(f"   Signal distribution: {dict(signal_counts)}")

        # Test returns
        data_with_returns = analyzer.calculate_returns(data_with_signals)
        print(f"✅ Returns calculated")

        # Test performance metrics
        metrics = analyzer.calculate_performance_metrics(data_with_returns)
        print(f"✅ Performance metrics calculated")
        print(f"   Total return: {metrics['total_return']*100:.2f}%")
        print(f"   Strategy return: {metrics['strategy_return']*100:.2f}%")
        print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.2f}")

        return data_with_returns, metrics

    except Exception as e:
        print(f"❌ Technical analysis error: {e}")
        return None, None


def test_visualization(data, symbol='SPY'):
    """Test visualization manually"""
    if data is None:
        print("⏭️  Skipping visualization (no data)")
        return

    print("\nTesting Visualization...")
    print("-" * 30)

    try:
        visualizer = Visualizer()
        chart_path = visualizer.plot_symbol_analysis(
            symbol, data, 'Test Symbol')

        if chart_path and os.path.exists(chart_path):
            print(f"✅ Chart created successfully: {chart_path}")
        else:
            print("❌ Chart creation failed")

    except Exception as e:
        print(f"❌ Visualization error: {e}")


def test_market_analyzer():
    """Test complete market analyzer"""
    print("\nTesting Complete Market Analyzer...")
    print("-" * 30)

    try:
        # Use single symbol for faster testing
        analyzer = MarketAnalyzer({'SPY': 'S&P 500 ETF'})

        # Test data fetching
        if not analyzer.fetch_data():
            print("❌ Data fetching failed")
            return
        print("✅ Data fetching successful")

        # Test analysis
        if not analyzer.run_analysis():
            print("❌ Analysis failed")
            return
        print("✅ Analysis successful")

        # Test outlook
        outlook = analyzer.get_market_outlook()
        print(f"✅ Market outlook: {outlook}")

        print("✅ Complete workflow successful!")

    except Exception as e:
        print(f"❌ Market analyzer error: {e}")


def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking Dependencies...")
    print("-" * 30)

    required_packages = {
        'yfinance': 'yf',
        'pandas': 'pd',
        'matplotlib': 'plt',
        'numpy': 'np'
    }

    missing_packages = []

    for package_name, import_name in required_packages.items():
        try:
            if import_name == 'yf':
                import yfinance as yf
            elif import_name == 'pd':
                import pandas as pd
            elif import_name == 'plt':
                import matplotlib.pyplot as plt
            elif import_name == 'np':
                import numpy as np
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            missing_packages.append(package_name)

    if missing_packages:
        print(f"\n⚠️  Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("✅ All dependencies installed")
    return True


def main():
    """Run manual tests step by step"""
    print("Market Analyzer Manual Test Suite")
    print("="*50)

    # Setup logging
    setup_logging('INFO')

    # Check dependencies first
    if not check_dependencies():
        return

    # Test individual components
    data = test_data_fetcher()
    analyzed_data, metrics = test_technical_analysis(data)
    test_visualization(analyzed_data)

    # Test complete workflow
    test_market_analyzer()

    print("\n" + "="*50)
    print("Manual testing completed!")
    print("="*50)


if __name__ == "__main__":
    main()

# ===================================

# tests/quick_test.py
"""Quick smoke test to verify basic functionality"""

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quick_test():
    """Quick test to verify imports and basic functionality"""
    print("Quick Smoke Test")
    print("="*30)

    try:
        # Test imports
        print("Testing imports...")
        from config import SYMBOLS, ANALYSIS_PERIOD
        from data_fetcher import DataFetcher
        from technical_analysis import TechnicalAnalyzer
        from visualization import Visualizer
        from market_analyzer import MarketAnalyzer
        from utils import setup_logging
        print("✅ All imports successful")

        # Test class instantiation
        print("Testing class instantiation...")
        fetcher = DataFetcher()
        analyzer = TechnicalAnalyzer()
        visualizer = Visualizer()
        market_analyzer = MarketAnalyzer()
        print("✅ All classes instantiated successfully")

        # Test configuration
        print("Testing configuration...")
        print(f"   Symbols configured: {len(SYMBOLS)}")
        print(f"   Analysis period: {ANALYSIS_PERIOD}")
        print("✅ Configuration loaded successfully")

        print("\n🎉 Quick test PASSED! Your refactored code structure is working.")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all files are in the correct locations")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    quick_test()

# ===================================

# Makefile (optional - for easy testing commands)
"""
# Makefile content (save as 'Makefile' without .py extension)

.PHONY: test quick-test unit-test integration-test clean

# Quick smoke test
quick-test:
	python tests/quick_test.py

# Run manual tests
manual-test:
	python tests/manual_test.py

# Run unit tests
unit-test:
	python -m pytest tests/test_*.py -v

# Run all tests
test:
	python tests/test_runner.py

# Clean generated files
clean:
	rm -rf output/
	rm -rf logs/
	rm -rf __pycache__/
	rm -rf tests/__pycache__/
	find . -name "*.pyc" -delete

# Install dependencies
install:
	pip install -r requirements.txt

# Run the main application
run:
	python main.py
"""
