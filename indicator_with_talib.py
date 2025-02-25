import pandas as pd
import numpy as np
import talib  

# Load your data
df = pd.read_excel('stock_ohlc_past_21_days.xlsx')

df_stock = df['Stock'].unique().tolist()
window = 3

# Create a list to store processed data
final_data = []

for i in df_stock: 
    stock_data = df[df['Stock'] == i].copy()
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.sort_values(by=['Date'], inplace=True) 

    # Calculate Indicators using custom functions
    stock_data['SMA9'] = talib.SMA(stock_data['Close'], timeperiod=9)
    stock_data['SMA21'] = talib.SMA(stock_data['Close'], timeperiod=21)
    if stock_data.shape[0] >= 14:
        stock_data['RSI14'] = talib.RSI(stock_data['Close'], timeperiod=14)
    else:
        stock_data['RSI14'] = np.nan  # Assign NaN instead of 0 to avoid logical errors

    stock_data['Swing_High'] = np.where(
        stock_data['High'] == stock_data['High'].rolling(window=window * 2 + 1, center=True).max(),
        stock_data['High'], np.nan
    )
    stock_data['Swing_Low'] = np.where(
        stock_data['Low'] == stock_data['Low'].rolling(window=window * 2 + 1, center=True).min(),
        stock_data['Low'], np.nan
    )

    # # Fill Swing High/Low for all rows
    # stock_data['Swing_High'].fillna(method='ffill', inplace=True)
    # stock_data['Swing_Low'].fillna(method='ffill', inplace=True)

    # Fill Swing High/Low for All Rows
    stock_data['Swing_High_Full'] = stock_data['Swing_High'].ffill().bfill()
    stock_data['Swing_Low_Full'] = stock_data['Swing_Low'].ffill().bfill()

    stock_data.drop(columns=['Swing_High', 'Swing_Low'], inplace=True)
    stock_data.rename(columns={'Swing_High_Full': 'Swing_High', 'Swing_Low_Full': 'Swing_Low'}, inplace=True)

    last_row = stock_data.iloc[-1].copy()

    # Buy/Sell Signal (fixing the values access issue)
    last_row['Buy_Signal'] = "TRUE" if (last_row['SMA9'] > last_row['SMA21']) & (last_row['RSI14'] > 50) else "FALSE"
    last_row['Sell_Signal'] = "TRUE" if (last_row['SMA9'] < last_row['SMA21']) & (last_row['RSI14'] < 50) else "FALSE"

    # Add Stop Loss based on Swing Levels
    last_row['Buy_Stop_Loss'] = last_row['Swing_Low'] if last_row['Buy_Signal'] == "TRUE" else 0
    last_row['Sell_Stop_Loss'] = last_row['Swing_High'] if last_row['Sell_Signal'] == "TRUE" else 0

    RRR = 2  # Risk-to-Reward Ratio of 2:1

    if last_row['Buy_Signal'] == "TRUE":
        last_row['Buy_Take_Profit'] = last_row['Close'] + (last_row['Close'] - last_row['Buy_Stop_Loss']) * RRR
    else:
        last_row['Buy_Take_Profit'] = 0

    if last_row['Sell_Signal'] == "TRUE":
        last_row['Sell_Take_Profit'] = last_row['Close'] - (last_row['Sell_Stop_Loss'] - last_row['Close']) * RRR
    else:
        last_row['Sell_Take_Profit'] = 0
        
    print(last_row)
    
    final_data.append(last_row)

# Convert list to DataFrame
final_df = pd.DataFrame(final_data)

# Save the final DataFrame to an Excel file
final_df.to_excel('stock_signals_with_talib_library.xlsx', index=False)

print("Stock signals saved to 'stock_signals_with_talib_library.xlsx'")

import pandas as pd

# Load the final sentiment analysis output
analyzer_df = pd.read_excel('final_output_with_ohlc.xlsx')

# Load the most recent stock data with indicators
indicators_df = pd.read_excel('stock_signals_with_talib_library.xlsx')

# Select only necessary columns from indicators data
indicator_columns = [
    'Stock', 'SMA9', 'SMA21', 'RSI14', 'Buy_Signal', 'Sell_Signal',
    'Swing_High', 'Swing_Low', 'Buy_Stop_Loss', 'Sell_Stop_Loss',
    'Buy_Take_Profit', 'Sell_Take_Profit'
]

indicators_df = indicators_df[indicator_columns]

# Merge the two dataframes based on 'Stock'
merged_df = pd.merge(analyzer_df, indicators_df, on='Stock', how='left')

# Save the merged dataframe to a new Excel file
merged_df.to_excel('final_output_with_talib_indicators.xlsx', index=False)

print("Final output with indicators merged successfully!")
