import os
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode, urlparse, parse_qs
from datetime import datetime, timedelta
from newspaper import Article
import pandas as pd
import yfinance as yf


import os
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



import custom_indicator


def fetch_stock_data():
    stock = input("Enter stock symbol: ").strip().upper()
    
    GOOGLE_NEWS_URL = "https://www.google.com/search"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    OUTPUT_DIR = "news_articles"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    FETCH_LIMIT = 15
    MAX_VALID_ARTICLES = 5

    def fetch_full_article(url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except:
            return ""

    def clean_google_url(google_url):
        parsed_url = urlparse(google_url)
        query_params = parse_qs(parsed_url.query)
        return query_params.get("q", [google_url])[0]

    def convert_relative_date(relative_date):
        now = datetime.today()
        if "hour" in relative_date:
            num = int(relative_date.split()[0])
            return (now - timedelta(hours=num)).strftime("%Y-%m-%d")
        elif "day" in relative_date:
            num = int(relative_date.split()[0])
            return (now - timedelta(days=num)).strftime("%Y-%m-%d")
        elif "week" in relative_date:
            num = int(relative_date.split()[0])
            return (now - timedelta(weeks=num)).strftime("%Y-%m-%d")
        return now.strftime("%Y-%m-%d")

    def scrape_news(stock):
        print(f"\nFetching news for: {stock}...")
        valid_articles = []
        params = {"q": f"{stock} stock news", "tbm": "nws", "hl": "en", "gl": "us"}
        url = f"{GOOGLE_NEWS_URL}?{urlencode(params)}"
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print(f"Failed to fetch news for {stock}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        for article in soup.select(".SoaBEf")[:FETCH_LIMIT]:
            title_element = article.select_one(".nDgy9d")
            url_element = article.select_one("a")
            date_element = article.select_one(".OSrXXb")
            
            title = title_element.text.strip() if title_element else "No Title"
            url = url_element["href"] if url_element else ""
            raw_date = date_element.text.strip() if date_element else "Unknown Date"
            
            if url.startswith("/url?"):
                url = clean_google_url(url)

            full_article = fetch_full_article(url)
            if not full_article.strip():
                continue

            valid_articles.append({
                "title": title,
                "url": url,
                "date": convert_relative_date(raw_date),
                "full_article": full_article,
            })
            if len(valid_articles) >= MAX_VALID_ARTICLES:
                break

        return valid_articles

    def save_articles(stock, articles):
        for i, article in enumerate(articles):
            filename = f"{article['date']}_{stock}.txt"
            filepath = os.path.join(OUTPUT_DIR, filename)
            counter = 1
            while os.path.exists(filepath):
                filepath = os.path.join(OUTPUT_DIR, f"{article['date']}_{stock}_{counter}.txt")
                counter += 1

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Stock: {stock}\nDate: {article['date']}\nTitle: {article['title']}\nURL: {article['url']}\nFull Article:\n{article['full_article']}\n")
            print(f"Saved: {filepath}")
        return len(articles) > 0

    def fetch_recent_ohlc(stock):
        try:
            ticker = yf.Ticker(stock)
            df = ticker.history(period="1d")
            if df.empty:
                print(f"No OHLC data found for {stock}")
                return None
            return {
                "Stock": stock,
                "Open_Price": df.iloc[-1]["Open"],
                "Close_Price": df.iloc[-1]["Close"],
                "High_Price": df.iloc[-1]["High"],
                "Low_Price": df.iloc[-1]["Low"],
                "Volume": df.iloc[-1]["Volume"],
            }
        except Exception as e:
            print(f"Failed to fetch OHLC data for {stock}: {e}")
            return None

    def fetch_past_21_days_ohlc(stock):
        try:
            ticker = yf.Ticker(stock)
            df = ticker.history(period="21d")
            if df.empty:
                print(f"No past 21 days OHLC data found for {stock}")
                return pd.DataFrame()
            df["Stock"] = stock
            df.reset_index(inplace=True)
            df["Date"] = df["Date"].dt.tz_localize(None)
            return df[["Date", "Stock", "Open", "High", "Close", "Low", "Volume"]]
        except Exception as e:
            print(f"Failed to fetch past 21 days OHLC data for {stock}: {e}")
            return pd.DataFrame()

    articles = scrape_news(stock)
    saved_articles = save_articles(stock, articles) if articles else False
    
    all_ohlc_data = []
    past_21_days_ohlc_data = []
    
    if saved_articles:
        ohlc_data = fetch_recent_ohlc(stock)
        if ohlc_data:
            all_ohlc_data.append(ohlc_data)
        past_21_days_df = fetch_past_21_days_ohlc(stock)
        if not past_21_days_df.empty:
            past_21_days_ohlc_data.append(past_21_days_df)

    if all_ohlc_data:
        ohlc_df = pd.DataFrame(all_ohlc_data)
        ohlc_df.to_excel("stock_ohlc_data.xlsx", index=False)
        print("\nOHLC data saved to 'stock_ohlc_data.xlsx'")
    else:
        print("\nNo OHLC data to save.")

    if past_21_days_ohlc_data:
        past_21_days_df = pd.concat(past_21_days_ohlc_data, ignore_index=True)
        past_21_days_df.to_excel("stock_ohlc_past_21_days.xlsx", index=False)
        print("\nPast 21 days OHLC data saved to 'stock_ohlc_past_21_days.xlsx'")
    else:
        print("\nNo past 21 days OHLC data to save.")
    
    print("\nNews scraping and OHLC data fetching completed!")

fetch_stock_data()





analyzer = SentimentIntensityAnalyzer()

def perform_sentiment_analysis(article_text):
    vader_scores = analyzer.polarity_scores(article_text)
    return vader_scores['compound']

def parse_date(date_string):
    formats = ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y']
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    print(f"Invalid date format: {date_string}")
    return None

def assign_dynamic_weights(group):
    max_date = group['Date'].max()
    group['Time_Diff_Days'] = (max_date - group['Date']).dt.days

    min_weight = 0.1
    max_weight = 0.9
    decay_factor = 0.5

    group['Weight'] = max_weight * np.exp(-decay_factor * group['Time_Diff_Days'])
    group['Weight'] = group['Weight'].clip(lower=min_weight, upper=max_weight)
    group['Weight'] = group['Weight'] / group['Weight'].sum()

    return group

def process_files(input_folder, output_excel, ohlc_file, final_output_with_ohlc):
    data = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()

                stock = date = title = url = None
                full_article_lines = []
                collecting_article = False

                for line in lines:
                    if line.startswith("Stock:"):
                        stock = line.split(":", 1)[1].strip()
                    elif line.startswith("Date:"):
                        date = line.split(":", 1)[1].strip()
                    elif line.startswith("Title:"):
                        title = line.split(":", 1)[1].strip()
                    elif line.startswith("URL:"):
                        url = line.split(":", 1)[1].strip()
                    elif line.startswith("Full Article:"):
                        collecting_article = True
                    elif collecting_article:
                        full_article_lines.append(line.strip())

                full_article = "\n".join(full_article_lines).strip()

                if not full_article or len(full_article.split()) < 5:
                    print(f"Skipping file due to empty or short article: {filename}")
                    continue

                sentiment_score = perform_sentiment_analysis(full_article)

                date = parse_date(date)
                if not date:
                    print(f"Skipping file due to invalid date: {filename}")
                    continue

                if stock and date and title and url:
                    data.append({
                        'Date': date,
                        'Stock': stock,
                        'Title': title,
                        'URL': url,
                        'Sentiment_score': sentiment_score
                    })
                else:
                    print(f"Skipping file due to missing fields: {filename}")

    if data:
        df = pd.DataFrame(data)
        df.sort_values(by=["Stock", "Date"], ascending=[True, False], inplace=True)

        final_results = []

        for stock, group in df.groupby('Stock'):
            group = group.head(5).copy()
            group = assign_dynamic_weights(group)

            weighted_avg_score = (group['Sentiment_score'] * group['Weight']).sum()
            sentiment_std = group['Sentiment_score'].std() or 0

            if weighted_avg_score >= sentiment_std:
                final_label_std = 'Positive'
            elif weighted_avg_score <= -sentiment_std:
                final_label_std = 'Negative'
            else:
                final_label_std = 'Neutral'

            if weighted_avg_score > 0.4:
                final_label_threshold = 'Positive'
            elif weighted_avg_score < -0.4:
                final_label_threshold = 'Negative'
            else:
                final_label_threshold = 'Neutral'

            for _, row in group.iterrows():
                final_results.append({
                    'Date': row['Date'],
                    'Stock': stock,
                    'Title': row['Title'],
                    'URL': row['URL'],
                    'Positive_score': max(row['Sentiment_score'], 0),
                    'Negative_score': abs(min(row['Sentiment_score'], 0)),
                    'Sentiment_score': row['Sentiment_score'],
                    'Weightage': row['Weight'],
                    'Weighted_average_score': weighted_avg_score,
                    'Final_sentiment_label_on_std': final_label_std,
                    'Final_sentiment_label_on_threshold_value': final_label_threshold
                })

        final_df = pd.DataFrame(final_results)
        final_df['Date'] = final_df['Date'].dt.strftime('%d %b %Y %H:%M')
        final_df.to_excel(output_excel, index=False)

        print(f"Output saved to {output_excel}")

        # Load OHLC Data
        ohlc_df = pd.read_excel(ohlc_file)

        # Merge with Analysis Results on 'Stock' only
        merged_df = pd.merge(final_df, ohlc_df, on='Stock', how='left')

        # Fill missing values for each stock by forward and backward fill
        merged_df[['Open_Price', 'Close_Price', 'High_Price', 'Low_Price', 'Volume']] = merged_df.groupby('Stock')[
            ['Open_Price', 'Close_Price', 'High_Price', 'Low_Price', 'Volume']
        ].transform(lambda x: x.ffill().bfill())

        merged_df.to_excel(final_output_with_ohlc, index=False)
        print(f"Final output with OHLC data saved to '{final_output_with_ohlc}'")

    else:
        print("No valid data to process.")

input_folder = 'news_articles'
output_excel = 'filtered_output1.xlsx'
ohlc_file = 'stock_ohlc_data.xlsx'
final_output_with_ohlc = 'final_output_with_ohlc.xlsx'

process_files(input_folder, output_excel, ohlc_file, final_output_with_ohlc)






# Load your data
df = pd.read_excel('stock_ohlc_past_21_days.xlsx')

df_stock = df['Stock'].unique().tolist()
window = 3

final_data = []

for i in df_stock: 
    stock_data = df[df['Stock'] == i].copy()
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.sort_values(by=['Date'], inplace=True) 
    stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
    stock_data = stock_data.reset_index(drop=True)

    # Calculate Indicators using custom functions
    stock_data['SMA9'] = custom_indicator .SMA(stock_data['Close'], period=9)
    stock_data['SMA21'] = custom_indicator .SMA(stock_data['Close'], period=21)
    if stock_data.shape[0] >= 14:
        stock_data = custom_indicator.calculate_rsi(stock_data, column="Close", period=14)
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
final_df.to_excel('stock_signals_with_custom_indicator.xlsx', index=False)

print("Stock signals saved to 'stock_signals_with_custom_indicator.xlsx")



import pandas as pd

# Load the final sentiment analysis output
analyzer_df = pd.read_excel('final_output_with_ohlc.xlsx')

# Load the most recent stock data with indicators
indicators_df = pd.read_excel('stock_signals_with_custom_indicator.xlsx')

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
merged_df.to_excel('final_output_with_custom_indicators.xlsx', index=False)

print("Final output with indicators merged successfully!")
