import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import json
import yfinance as yf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import warnings
warnings.filterwarnings('ignore')


class KenyaStockScraperFixed:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        # Target stocks for analysis
        self.target_stocks = {
            'SCOM': 'Safaricom PLC',
            'EQTY': 'Equity Group Holdings',
            'KCB': 'KCB Group',
            'EABL': 'East African Breweries',
            'COOP': 'Co-operative Bank',
            'NCBA': 'NCBA Group',
            'BAT': 'British American Tobacco Kenya',
            'ABSA': 'Absa Bank Kenya'
        }

    def setup_selenium_driver(self):
        """Setup Chrome driver with proper options"""
        options = Options()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option(
            "excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        try:
            driver = webdriver.Chrome(options=options)
            driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except Exception as e:
            print(f"Chrome driver setup failed: {e}")
            return None

    def test_website_access(self):
        """Test access to key websites"""
        test_urls = [
            'https://www.nse.co.ke/',
            'https://www.investing.com/indices/kenya-nse-20-historical-data',
            'https://tradingeconomics.com/kenya/stock-market'
        ]

        results = {}
        for url in test_urls:
            try:
                response = self.session.get(url, timeout=10)
                results[url] = {
                    'status': response.status_code,
                    'accessible': response.status_code == 200,
                    'content_length': len(response.content)
                }
                print(f"✓ {url}: Status {response.status_code}")
            except Exception as e:
                results[url] = {'status': 'Error',
                                'accessible': False, 'error': str(e)}
                print(f"✗ {url}: {e}")

            time.sleep(1)

        return results

    def scrape_trading_economics_kenya(self):
        """
        Scrape Kenya stock market data from Trading Economics
        More reliable source with better structure
        """
        url = "https://tradingeconomics.com/kenya/stock-market"

        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract current NSE20 index value
            market_data = {}

            # Look for the main index value
            index_elements = soup.find_all(['span', 'div'], class_=[
                                           'te-value', 'index-value'])
            for elem in index_elements:
                if elem.text.strip().replace(',', '').replace('.', '').isdigit():
                    market_data['NSE20_Index'] = float(
                        elem.text.strip().replace(',', ''))
                    break

            # Look for percentage change
            change_elements = soup.find_all(
                ['span', 'div'], text=lambda t: t and '%' in str(t))
            for elem in change_elements[:3]:  # Check first 3 matches
                text = elem.text.strip()
                if '%' in text and any(char in text for char in ['+', '-']):
                    market_data['Daily_Change_Percent'] = text
                    break

            market_data['Last_Updated'] = datetime.now()
            market_data['Source'] = 'Trading Economics'

            print(f"Trading Economics data: {market_data}")
            return market_data

        except Exception as e:
            print(f"Error scraping Trading Economics: {e}")
            return {}

    def scrape_investing_com_manual(self):
        """
        Scrape Kenya NSE data from Investing.com with better error handling
        """
        url = "https://www.investing.com/indices/kenya-nse-20-historical-data"

        driver = self.setup_selenium_driver()
        if not driver:
            return pd.DataFrame()

        try:
            print("Loading Investing.com page...")
            driver.get(url)

            # Wait for page to load
            wait = WebDriverWait(driver, 20)

            # Look for historical data table
            try:
                # Try different table selectors
                table_selectors = [
                    'table[data-test="historical-data-table"]',
                    'table.freeze-column-w-1',
                    'table.genTbl',
                    'div[data-test="historical-data-table"] table',
                    '.historyDataTable table'
                ]

                table = None
                for selector in table_selectors:
                    try:
                        table = wait.until(EC.presence_of_element_located(
                            (By.CSS_SELECTOR, selector)))
                        print(f"Found table with selector: {selector}")
                        break
                    except:
                        continue

                if not table:
                    # Try to find any table on the page
                    tables = driver.find_elements(By.TAG_NAME, 'table')
                    if tables:
                        table = tables[0]  # Use the first table found
                        print(
                            f"Using first table found, total tables: {len(tables)}")

                if table:
                    rows = table.find_elements(By.TAG_NAME, 'tr')[
                        1:]  # Skip header

                    historical_data = []
                    # Limit to 100 rows for testing
                    for i, row in enumerate(rows[:100]):
                        try:
                            cols = row.find_elements(By.TAG_NAME, 'td')
                            if len(cols) >= 6:
                                # Extract data with better error handling
                                date_text = cols[0].text.strip()
                                price_text = cols[1].text.strip().replace(
                                    ',', '')

                                data_point = {
                                    'Date': date_text,
                                    'Price': price_text,
                                    'Open': cols[2].text.strip().replace(',', '') if len(cols) > 2 else '',
                                    'High': cols[3].text.strip().replace(',', '') if len(cols) > 3 else '',
                                    'Low': cols[4].text.strip().replace(',', '') if len(cols) > 4 else '',
                                    'Volume': cols[5].text.strip() if len(cols) > 5 else '',
                                    'Change_Percent': cols[6].text.strip() if len(cols) > 6 else ''
                                }
                                historical_data.append(data_point)
                        except Exception as e:
                            print(f"Error processing row {i}: {e}")
                            continue

                    if historical_data:
                        df = pd.DataFrame(historical_data)
                        print(f"Collected {len(df)} rows of historical data")
                        return df
                    else:
                        print("No data extracted from table")

            except Exception as e:
                print(f"Error finding historical data table: {e}")

        except Exception as e:
            print(f"Error accessing Investing.com: {e}")

        finally:
            if driver:
                driver.quit()

        return pd.DataFrame()

    def get_yahoo_finance_kenya_data(self):
        """
        Get Kenya market data from Yahoo Finance
        """
        # Try FTSE NSE Kenya 25 Index
        kenya_symbols = [
            'FNKEN2.L',  # FTSE NSE Kenya 25 Index
            '^NSE20',    # NSE 20 Index (if available)
        ]

        data_collected = {}

        for symbol in kenya_symbols:
            try:
                print(f"Trying Yahoo Finance symbol: {symbol}")
                ticker = yf.Ticker(symbol)

                # Get historical data for 5 years
                hist_data = ticker.history(period='5y')

                if not hist_data.empty:
                    hist_data.reset_index(inplace=True)
                    hist_data['Symbol'] = symbol
                    data_collected[symbol] = hist_data
                    print(f"✓ Collected {len(hist_data)} records for {symbol}")
                else:
                    print(f"✗ No data for {symbol}")

            except Exception as e:
                print(f"Error with Yahoo Finance {symbol}: {e}")

        return data_collected

    def download_mendeley_data(self):
        """
        Instructions for downloading research data from Mendeley
        """
        mendeley_links = {
            '2023-2024': 'https://data.mendeley.com/datasets/ss5pfw8xnk/1',
            '2022': 'https://data.mendeley.com/datasets/jmcdmnyh2s/2',
            '2013-2020': 'https://data.mendeley.com/datasets/73rb78pmzw/2'
        }

        print("\n=== MANUAL DATA DOWNLOAD OPTIONS ===")
        print("The following research datasets contain comprehensive NSE data:")

        for period, link in mendeley_links.items():
            print(f"\n{period} Data: {link}")

        print("\nThese datasets contain CSV files with historical data for all NSE stocks.")
        print("Download manually and place in 'manual_data/' folder for processing.")

        return mendeley_links

    def create_sample_data(self):
        """
        Create sample data structure for testing analysis code
        """
        print("Creating sample data for testing...")

        # Generate sample data for each target stock
        sample_data = {}

        for symbol, company in self.target_stocks.items():
            # Create 2 years of sample daily data
            dates = pd.date_range(end=datetime.now(), periods=1825, freq='D')

            # Generate realistic stock price movements
            # Reproducible but different for each stock
            np.random.seed(hash(symbol) % 1000)

            base_price = np.random.uniform(10, 100)  # Random base price
            returns = np.random.normal(
                0.001, 0.02, len(dates))  # Daily returns
            prices = [base_price]

            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            sample_df = pd.DataFrame({
                'Date': dates,
                'Close': prices,
                'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
                'High': [p * np.random.uniform(1.0, 1.05) for p in prices],
                'Low': [p * np.random.uniform(0.95, 1.0) for p in prices],
                'Volume': np.random.randint(10000, 1000000, len(dates)),
                'Symbol': symbol,
                'Company': company
            })

            sample_data[symbol] = sample_df

        print(f"Sample data created for {len(sample_data)} stocks")
        return sample_data

    def run_comprehensive_data_collection(self):
        """
        Run all data collection methods and return best available data
        """
        print("=== KENYA STOCK MARKET DATA COLLECTION ===\n")

        # Step 1: Test website accessibility
        print("1. Testing website access...")
        access_results = self.test_website_access()

        # Step 2: Try Trading Economics for current market data
        print("\n2. Getting current market data from Trading Economics...")
        current_data = self.scrape_trading_economics_kenya()

        # Step 3: Try Yahoo Finance for historical data
        print("\n3. Attempting Yahoo Finance data collection...")
        yahoo_data = self.get_yahoo_finance_kenya_data()

        # Step 4: Try Investing.com for historical data
        print("\n4. Attempting Investing.com historical data...")
        investing_data = self.scrape_investing_com_manual()

        # Step 5: Show manual download options
        print("\n5. Manual data download options...")
        mendeley_links = self.download_mendeley_data()

        # Step 6: Create sample data for testing
        print("\n6. Creating sample data for analysis testing...")
        sample_data = self.create_sample_data()

        # Compile results
        results = {
            'website_access': access_results,
            'current_market_data': current_data,
            'yahoo_finance_data': yahoo_data,
            'investing_historical': investing_data,
            'manual_download_links': mendeley_links,
            'sample_data': sample_data
        }

        return results

    def save_collected_data(self, results, output_dir='kenya_stock_data'):
        """
        Save all collected data to files
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save current market data
        if results['current_market_data']:
            with open(f'{output_dir}/current_market_data.json', 'w') as f:
                json.dump(results['current_market_data'],
                          f, indent=2, default=str)

        # Save Yahoo Finance data
        if results['yahoo_finance_data']:
            for symbol, df in results['yahoo_finance_data'].items():
                df.to_csv(f'{output_dir}/yahoo_{symbol}.csv', index=False)

        # Save Investing.com data
        if not results['investing_historical'].empty:
            results['investing_historical'].to_csv(
                f'{output_dir}/investing_nse20_historical.csv', index=False)

        # Save sample data
        if results['sample_data']:
            for symbol, df in results['sample_data'].items():
                df.to_csv(f'{output_dir}/sample_{symbol}.csv', index=False)

        # Save manual download links
        with open(f'{output_dir}/manual_data_sources.json', 'w') as f:
            json.dump(results['manual_download_links'], f, indent=2)

        # Create summary report
        summary = {
            'collection_date': datetime.now().isoformat(),
            'websites_accessible': sum(1 for r in results['website_access'].values() if r.get('accessible', False)),
            'yahoo_finance_symbols': list(results['yahoo_finance_data'].keys()) if results['yahoo_finance_data'] else [],
            'investing_rows_collected': len(results['investing_historical']) if not results['investing_historical'].empty else 0,
            'sample_data_stocks': len(results['sample_data']) if results['sample_data'] else 0,
            'next_steps': [
                "Review manually downloaded data from Mendeley links",
                "Use sample data to test analysis algorithms",
                "Consider paid data sources for real-time feeds",
                "Set up automated daily data collection"
            ]
        }

        with open(f'{output_dir}/collection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== DATA COLLECTION COMPLETE ===")
        print(f"Results saved to: {output_dir}/")
        print(f"Summary: {summary}")

        return summary


# Usage
if __name__ == "__main__":
    scraper = KenyaStockScraperFixed()

    # Run comprehensive data collection
    results = scraper.run_comprehensive_data_collection()

    # Save all results
    summary = scraper.save_collected_data(results)

    print("\n=== RECOMMENDATIONS ===")
    print("1. For immediate analysis: Use the sample data generated")
    print("2. For comprehensive historical data: Download from Mendeley links")
    print("3. For current prices: Use the Trading Economics scraper")
    print("4. Consider setting up a paid NSE data subscription for real-time data")
