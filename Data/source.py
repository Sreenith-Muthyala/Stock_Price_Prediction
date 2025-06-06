import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df