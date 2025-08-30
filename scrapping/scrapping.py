import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def analyze_us_markets():
    """Analyze US markets which are easily accessible"""
    print("Analyzing US Market Data...")

    # US ETFs and stocks (easily accessible)
    symbols = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'NASDAQ-100 ETF',
        'DIA': 'Dow Jones ETF',
        'IWM': 'Russell 2000 ETF',
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corp.'
    }

    results = {}

    for symbol, name in symbols.items():
        try:
            # Download 5 years of data with auto_adjust explicitly set
            data = yf.download(symbol, period='5y', auto_adjust=True)

            if not data.empty and len(data) > 200:  # Ensure enough data
                # Calculate moving averages
                data['MA50'] = data['Close'].rolling(window=50).mean()
                data['MA200'] = data['Close'].rolling(window=200).mean()

                # Generate signals properly
                signal_condition = data['MA50'] > data['MA200']
                data['Signal'] = signal_condition.astype(int)
                data.loc[data['MA50'].isna() | data['MA200'].isna(),
                         'Signal'] = 0

                data['Position'] = data['Signal'].diff()

                # Calculate returns
                data['Daily_Return'] = data['Close'].pct_change()
                data['Strategy_Return'] = data['Signal'].shift(
                    1) * data['Daily_Return']

                # Calculate performance metrics safely
                strategy_returns = data['Strategy_Return'].dropna()

                # Ensure we're getting scalar values
                total_return_val = float(
                    data['Close'].iloc[-1] / data['Close'].iloc[0] - 1)

                if len(strategy_returns) > 0:
                    strategy_cum_return = float(
                        (1 + strategy_returns).cumprod().iloc[-1] - 1)
                    if strategy_returns.std() > 0:
                        sharpe_ratio_val = float(
                            strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))
                    else:
                        sharpe_ratio_val = 0.0
                else:
                    strategy_cum_return = 0.0
                    sharpe_ratio_val = 0.0

                # Store results as simple values, not Series
                results[symbol] = {
                    'data': data,
                    'total_return': total_return_val,
                    'strategy_return': strategy_cum_return,
                    'sharpe_ratio': sharpe_ratio_val,
                    'name': name
                }

                print(f"✓ {symbol} ({name}): {len(data)} records")

        except Exception as e:
            print(f"✗ Error with {symbol}: {e}")

    return results


def generate_trading_signals(results):
    """Generate buy/sell signals based on analysis"""
    print("\n" + "="*60)
    print("TRADING SIGNALS ANALYSIS - MOVING AVERAGE CROSSOVER STRATEGY")
    print("="*60)

    for symbol, result in results.items():
        data = result['data']

        # Get recent signals
        recent_data = data.tail(10)
        current_signal = int(recent_data['Signal'].iloc[-1])
        previous_signal = int(recent_data['Signal'].iloc[-2])

        # Get current price safely
        current_price = float(data['Close'].iloc[-1])
        ma50 = float(data['MA50'].iloc[-1])
        ma200 = float(data['MA200'].iloc[-1])

        print(f"\n{symbol} ({result['name']}):")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  50-Day MA: ${ma50:.2f}")
        print(f"  200-Day MA: ${ma200:.2f}")
        print(f"  Trend: {'BULLISH' if current_signal == 1 else 'BEARISH'}")

        if current_signal != previous_signal:
            action = "BUY" if current_signal == 1 else "SELL"
            print(f"  ⚡ NEW SIGNAL: {action}")

        # These are now guaranteed to be floats, not Series
        print(f"  Buy & Hold Return: {result['total_return']*100:+.2f}%")
        print(f"  Strategy Return: {result['strategy_return']*100:+.2f}%")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")


def plot_results(results):
    """Plot the analysis results"""
    for symbol, result in results.items():
        data = result['data']

        plt.figure(figsize=(14, 10))

        # Plot price and moving averages
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'],
                 label='Price', linewidth=2, color='black')
        plt.plot(data.index, data['MA50'],
                 label='50-Day MA', linewidth=1.5, color='blue')
        plt.plot(data.index, data['MA200'],
                 label='200-Day MA', linewidth=1.5, color='red')

        # Plot buy/sell signals
        buy_signals = data[data['Position'] == 1]
        sell_signals = data[data['Position'] == -1]

        if not buy_signals.empty:
            plt.scatter(buy_signals.index, buy_signals['Close'],
                        color='green', marker='^', s=100, label='Buy Signal', alpha=0.7)
        if not sell_signals.empty:
            plt.scatter(sell_signals.index, sell_signals['Close'],
                        color='red', marker='v', s=100, label='Sell Signal', alpha=0.7)

        plt.title(f'{symbol} - Price and Moving Average Crossover Strategy',
                  fontsize=14, fontweight='bold')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot cumulative returns
        plt.subplot(2, 1, 2)
        cumulative_returns = (1 + data['Daily_Return']).cumprod()
        strategy_returns = (1 + data['Strategy_Return'].fillna(0)).cumprod()

        plt.plot(data.index, cumulative_returns,
                 label='Buy & Hold', linewidth=2, color='blue')
        plt.plot(data.index, strategy_returns,
                 label='MA Crossover Strategy', linewidth=2, color='green')
        plt.title('Cumulative Returns Comparison',
                  fontsize=14, fontweight='bold')
        plt.ylabel('Growth of $1')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{symbol}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Chart saved: {symbol}_analysis.png")


def print_summary_statistics(results):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (5-Year Performance)")
    print("="*60)

    summary_data = []
    for symbol, result in results.items():
        summary_data.append({
            'Symbol': symbol,
            'Buy&Hold (%)': result['total_return'] * 100,
            'Strategy (%)': result['strategy_return'] * 100,
            'Sharpe Ratio': result['sharpe_ratio'],
            'Outperformance': (result['strategy_return'] - result['total_return']) * 100
        })

    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('Sharpe Ratio', ascending=False)

    print(df_summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    best_strategy = df_summary.iloc[0]
    print(f"\n🏆 Best Performing Strategy: {best_strategy['Symbol']}")
    print(f"   Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")
    print(f"   Strategy Return: {best_strategy['Strategy (%)']:.2f}%")


def main():
    """Main analysis function"""
    print("Starting Stock Market Analysis...")
    print("="*60)
    print("Strategy: Moving Average Crossover (50-day vs 200-day)")
    print("="*60)

    # Analyze US markets
    results = analyze_us_markets()

    if results:
        # Generate trading signals
        generate_trading_signals(results)

        # Print summary statistics
        print_summary_statistics(results)

        # Plot results
        plot_results(results)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print("📈 Charts saved as high-resolution PNG files")
        print("💡 Trading signals generated based on MA crossover")
        print("📊 Performance statistics calculated")
        print("="*60)

        # Show current market status
        print("\nCURRENT MARKET OUTLOOK:")
        bullish_count = sum(1 for result in results.values()
                            if result['data']['Signal'].iloc[-1] == 1)
        total_count = len(results)
        print(f"📊 {bullish_count}/{total_count} assets in BULLISH trend")

    else:
        print("No data could be analyzed. Please check your internet connection.")


if __name__ == "__main__":
    main()
