import yfinance as yf
import pandas as pd
import numpy as np

def get_data(ticker, start_date, end_date, close=True):
    if close:
        data = pd.DataFrame(yf.download(ticker, start_date, end_date)['Adj Close'])

        # convert to log returns, dropna
        data = data.apply(lambda x: np.log(x / x.shift(1))).dropna()

        # change the data shape to (n // 5, 5)
        if len(data) // 5 != 0:
            data = data.iloc[:len(data) // 5 * 5]
            data = data.values.reshape(len(data) // 5, 5)
        else:
            data = data.values.reshape(1, 5)

        return data

    else:
        data =  yf.download(ticker, start_date, end_date)
        close = data['Adj Close']
        close = close.apply(lambda x: np.log(x / x.shift(1)))
        data['Adj Close'] = close

        data = data.dropna(inplace=True)

        return data
