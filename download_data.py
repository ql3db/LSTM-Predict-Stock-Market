# pip install yfinance pandas
# pip3 install torch torchvision torchaudio
import pandas as pd
import yfinance as yf
import yahooquery as yq
from fredapi import Fred

########
#Dowload GICS sectors
########
def fetch_sp500_info():
    """ Fetches the S&P 500 stock tickers and their corresponding industry from the Wikipedia page """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url, header=0)[0]
    sp500_table['Symbol'] = sp500_table['Symbol'].str.replace('.', '-')  # Adjusting ticker symbols
    return sp500_table[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]

# Fetch S&P 500 company tickers and industries
sp500_info = fetch_sp500_info()

# Save the data to a CSV file
# sp500_info.to_csv("sp500_tickers_and_industries.csv", index=False)


#######
# Download stock prices for stocks
######
def download_stock_data(tickers):
    """ Downloads historical stock data for the given tickers """
    data = yf.download(tickers, start="2019-12-31", end="2024-05-10")  # Customize dates as needed
    return data['Adj Close']  # Adjusted Close prices

# Fetch S&P 500 company tickers and sectors
tickers = sp500_info['Symbol'].str.replace('.', '-').tolist()  # Adjusting ticker symbols, Yahoo Finance uses '-' instead of '.'
tickers.append('DX-Y.NYB')
tickers.append('^TNX')
# Download the stock data
stock_data = download_stock_data(tickers)

# Save the data to a CSV file
stock_data.to_csv("sp500_stock_prices.csv")

######
#Download Macro Data
######
fred = Fred(api_key='747d5c0fab23a328c7fefa3f91af25c2')

def get_data(series_id):
    """Fetches data for a given FRED series ID."""
    data = fred.get_series(series_id)
    return data

# FRED series IDs for Treasury rates
series_ids = {
    # '10Y Rate': 'DGS10',  # Daily 10-Year Treasury Constant Maturity Rate
    # '3M Rate': 'DTB3',   # Daily Treasury Bill Rates: 3-Month
    # '5Y Rate': 'DGS5',    # Daily 5-Year Treasury Constant Maturity Rate
    # '30Y Rate': 'DGS30',   # Daily 30-Year Treasury Constant Maturity Rate
    'CPI': 'CPIAUCSL',
    'Core CPI': 'CORESTICKM159SFRBATL', # Core CPI
    'GDP': 'GDP',
    'Dollar Index': 'DTWEXM',   # Trade Weighted U.S. Dollar Index: Major Currencies
    'Imports': 'IEABC',        # U.S. Imports of Goods and Services, Balance of Payments Basis (Millions of Dollars)
    'Exports': 'EXPGSC1',      # U.S. Exports of Goods and Services, Balance of Payments Basis (Millions of Dollars)
    'PCE': 'PCE',  # Personal Consumption Expenditures
    'Housing Starts': 'HOUST',       # New Privately-Owned Housing Units Started: Total Units
    'Unemployment Rate': 'UNRATE',   # Civilian Unemployment Rate
    'Job Openings': 'JTSJOL'
}

# Download data
macro_data = {name: get_data(series) for name, series in series_ids.items()}

# Convert the data to a DataFrame for easier manipulation and save it
macro_data_df = pd.DataFrame(macro_data)
print(macro_data_df.columns)
macro_data_df.to_csv('macro_data.csv')

#####
#Download type and market cap data
#####
# Function to get market cap
def get_market_cap(symbol):
    try:
        stock = yf.Ticker(symbol)
        market_cap = stock.info['marketCap']
        return market_cap
    except:
        return None

# Function to download tickers from ETF holdings
growth_tickers = [
    'MSFT', 'NVDA', 'AAPL', 'AMZN', 'META', 'GOOGL', 'GOOG', 'LLY', 'AVGO', 'TSLA',
    'NFLX', 'V', 'AMD', 'MA', 'CRM', 'ADBE', 'ORCL', 'AMAT', 'UNH', 'INTU',
    'COST', 'NOW', 'PG', 'UBER', 'BKNG', 'MRK', 'LRCX', 'HD', 'QCOM', 'LIN',
    'KLAC', 'ABBV', 'CAT', 'ACN', 'PANW', 'COP', 'AXP', 'KO', 'ISRG', 'SNPS',
    'PEP', 'CMG', 'TMO', 'ETN', 'CDNS', 'MCD', 'GE', 'ANET', 'TJX', 'SYK',
    'TDG', 'VRTX', 'ADI', 'TXN', 'UNP', 'REGN', 'SPGI', 'PGR', 'BX', 'BSX',
    'MAR', 'PM', 'DE', 'MMC', 'PH', 'SBUX', 'NXPI', 'TT', 'HLT', 'NKE',
    'EOG', 'ROST', 'CPRT', 'APH', 'FI', 'BA', 'DHI', 'SMCI', 'URI', 'ZTS',
    'ORLY', 'PCAR', 'HES', 'FTNT', 'FCX', 'MDLZ', 'ADP', 'ACGL', 'RCL', 'SHW',
    'VST', 'WM', 'AMT', 'FANG', 'MPWR', 'MSI', 'LULU', 'MPC', 'FICO', 'EQIX',
    'IT', 'CTAS', 'CSX', 'CEG', 'ABNB', 'AZO', 'CL', 'ECL', 'GWW', 'HCA',
    'ITW', 'DXCM', 'MCO', 'MCHP', 'OKE', 'ODFL', 'PWR', 'AON', 'AJG', 'IR',
    'DECK', 'CME', 'CARR', '--', 'FTV', 'NUE', 'WST', 'TRGP', 'PHM', 'ROP',
    'VMC', 'NVR', 'MNST', 'MLM', 'LEN', 'MSCI', 'IDXX', 'AMP', 'ADSK', 'AXON',
    'BLDR', 'CPAY', 'EW', 'HWM', 'GEV', 'VRSK', 'WMB', 'TYL', 'RSG', 'SPG',
    'PSA', 'YUM', 'TEL', 'FAST', 'DAL', 'DLR', 'DFS', 'AME', 'ANSS', 'BRO',
    'EA', 'CSGP', 'CE', 'EXPE', 'IQV', 'PAYX', 'OTIS', 'ROK', 'STLD', 'PTC',
    'ON', 'TSCO', 'ULTA', 'TTWO', 'GRMN', 'IRM', 'LYV', 'MTD', 'CTRA', 'CHTR',
    'CHD', 'BR', 'CBOE', 'CDW', 'STE', 'ALGN', 'APA', 'CCL', 'DRI', 'MOH',
    'NTAP', 'MRO', 'LVS', 'EFX', 'HSY', 'HUBB', 'PNR', 'WYNN', 'VRSN', 'WAT',
    'POOL', 'SBAC', 'AOS', 'SNA', 'NCLH', 'STX', 'HST', 'EXPD', 'FDS', 'LW',
    'JBL', 'MGM', 'MAS', 'ENPH', 'DPZ', 'COO', 'AKAM', 'CF', 'DAY', 'CZR',
    'EPAM', 'DVA', 'PODD', 'GNRC', 'ROL', 'RL', 'ALLE', 'PAYC', 'ETSY', 'XOM',
    'HRL'
]
value_tickers = [
    'BRK/B', 'JPM', 'XOM', 'JNJ', 'UNH', 'WMT', 'CVX', 'BAC', 'PG', 'WFC',
    'COST', 'HD', 'MRK', 'CSCO', 'DIS', 'ABT', 'ABBV', 'DHR', 'VZ', 'NEE',
    'AMGN', 'PFE', 'IBM', 'CMCSA', 'PEP', 'GS', 'KO', 'MU', 'V', 'RTX',
    'TMO', 'HON', 'INTC', 'LOW', 'MS', 'T', 'ELV', 'C', 'TXN', 'QCOM',
    'MDT', 'CB', 'BLK', 'SCHW', 'MCD', 'MA', 'GE', 'LMT', 'CI', 'PM',
    'LIN', 'PLD', 'UPS', 'TMUS', 'ACN', 'SO', 'BMY', 'MO', 'GILD', 'DUK',
    'SPGI', 'UNP', 'ICE', 'MCK', 'CAT', 'CVS', 'GD', 'TGT', 'SLB',
    'PYPL', 'BDX', 'EMR', 'NKE', 'NOC', 'USB', 'PSX', 'PNC', 'PGR', 'ADP',
    'APD', 'BA', 'FDX', 'WELL', 'MMM', 'AIG', 'COF', 'MDLZ', 'VLO', 'TFC',
    'ETN', 'NSC', 'GM', 'AMT', 'BSX', 'CME', 'MRNA', 'NEM', 'MMC',
    'ISRG', 'JCI', 'TRV', 'SRE', 'AEP', 'ADI', 'AFL', 'CL', 'D', 'BK',
    'FIS', 'F', 'FI', 'HUM', 'KMB', 'MET', 'A', 'ALL', 'CCI', 'COP',
    'PRU', 'O', 'SYK', 'TJX', 'WM', 'VRTX', 'REGN', 'OXY', 'DE', 'DOW',
    'AXP', 'LHX', 'GIS', 'FCX', 'EQIX', 'CTVA', 'STZ', 'CEG', 'CNC', 'PCG',
    'PEG', 'SBUX', 'CMI', 'COR', 'EXC', 'KMI', 'KVUE', 'KDP', 'KR', 'ITW',
    'APH', 'DD', 'SYY', 'ZTS', 'SHW', 'ROP', 'XYL', 'EW', 'CTSH', 'ADM',
    'ABNB', 'BKR', 'CSX', 'HCA', 'HAL', 'GEHC', 'MCO', 'BIIB', 'ED',
    'DVN', 'DG', 'WAB', 'RMD', 'SPG', 'PPG', 'VICI', 'WMB', 'XEL', 'ECL',
    'EIX', 'MPC', 'EL', 'EXR', 'FSLR', 'HIG', 'HPQ', 'GPN', 'AJG', 'GEV',
    'EBAY', 'KHC', 'GLW', 'CARR', 'AVB', 'AON', 'TEL', 'LYB', 'WTW', 'WEC',
    'WDC', 'TROW', 'PSA', 'CBRE', 'AWK', 'CTAS', 'EOG', 'DOV', 'DLR',
    'KEYS', 'MTB', 'IQV', 'IFF', 'MSI', 'FITB', 'HPE', 'NDAQ', 'DLTR',
    'DTE', 'ETR', 'CAH', 'BX', 'RJF', 'PAYX', 'OTIS', 'ZBH', 'STT', 'TT',
    'APTV', 'TER', 'VLTO', 'WY', 'YUM', 'PPL', 'BALL', 'ADSK', 'MCHP',
    'EQR', 'ES', 'FAST', 'FE', 'HBAN', 'GPC', 'NRG', 'MTD', 'INVH', 'LDOS',
    'AMP', 'AME', 'AEE', 'CNP', 'CSGP', 'DXCM', 'VTR', 'UAL', 'TSN',
    'SYF', 'TSCO', 'TDY', 'TXT', 'WBD', 'EG', 'PFG', 'OMC', 'ON', 'OKE',
    'NTRS', 'RF', 'NXPI', 'EA', 'EQT', 'CINF', 'ATO', 'ARE', 'AVY', 'CDW',
    'CMS', 'BAX', 'LEN', 'IDXX', 'J', 'MKC', 'MSCI', 'HOLX', 'HSY', 'EFX',
    'ESS', 'IEX', 'ILMN', 'MAA', 'NTAP', 'MNST', 'IP', 'LH', 'K', 'AZO',
    'WRB', 'ALB', 'CFG', 'CLX', 'BG', 'RSG', 'DGX'
]
#remove the DXY and 10year rate in tickers
tickers = tickers[:-2]
overlapping_tickers = list(set(growth_tickers) & set(value_tickers))
growth_tickers = [ticker for ticker in growth_tickers if ticker not in overlapping_tickers]
value_tickers = [ticker for ticker in value_tickers if ticker not in overlapping_tickers]
blend_tickers = [ticker for ticker in tickers if (ticker not in growth_tickers)&(ticker not in value_tickers)]
# Create DataFrames for each type
growth_df = pd.DataFrame({'Ticker': growth_tickers, 'Type': 'Growth'})
value_df = pd.DataFrame({'Ticker': value_tickers, 'Type': 'Value'})
blend_df = pd.DataFrame({'Ticker': blend_tickers, 'Type': 'Blend'})
# Concatenate the DataFrames
type_df = pd.concat([growth_df, value_df, blend_df], ignore_index=True)
# Fetch market caps
market_caps = {}
for symbol in tickers:
    market_cap = get_market_cap(symbol)
    if market_cap:
        market_caps[symbol] = market_cap
# Convert to DataFrame
market_cap_df = pd.DataFrame(list(market_caps.items()), columns=['Ticker', 'Market Cap'])
type_market_cap_df = pd.merge(type_df,market_cap_df,on='Ticker',how='right')
# Calculate Percentile Ranks
type_market_cap_df['Percentile'] = type_market_cap_df['Market Cap'].rank(pct=True)
# Define the categorize function
def categorize_cap(row):
    if row['Percentile'] > 0.6667:
        return 'Large Cap'
    elif row['Percentile'] > 0.3333:
        return 'Mid Cap'
    else:
        return 'Small Cap'
# Apply the categorize function
type_market_cap_df['Cap Category'] = type_market_cap_df.apply(categorize_cap, axis=1)
# Add a new column combining 'Type' and 'Cap Category'
type_market_cap_df['Type_Cap_Category'] = type_market_cap_df['Type'] + ' ' + type_market_cap_df['Cap Category']
type_market_cap_df.to_csv('sp500_type_market_cap.csv', index=False)



