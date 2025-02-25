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

# Stocks to scrape news for
stocks = [ "INFY"# "SPG", "WELL"
    # "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "VLO", "MPC", "PSX",
    # "DOW", "DD", "LIN", "APD", "SHW", "NEM", "FCX", "ALB", "IP", "ECL",
    # "GE", "HON", "UNP", "BA", "MMM", "CAT", "LMT", "RTX", "NOC", "DE",
    # "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "BKNG", "LOW", "TGT", "GM"
]

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
    except Exception as e:
        return ""


def clean_google_url(google_url):
    parsed_url = urlparse(google_url)
    query_params = parse_qs(parsed_url.query)
    return query_params.get("q", [google_url])[0]


def convert_relative_date(relative_date):
    now = datetime.today()
    if "hour" in relative_date:
        num = int(relative_date.split()[0])
        article_date = now - timedelta(hours=num)
    elif "day" in relative_date:
        num = int(relative_date.split()[0])
        article_date = now - timedelta(days=num)
    elif "week" in relative_date:
        num = int(relative_date.split()[0])
        article_date = now - timedelta(weeks=num)
    elif "minute" in relative_date:
        article_date = now
    else:
        article_date = now
    return article_date.strftime("%Y-%m-%d")


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
    articles_fetched = 0

    for article in soup.select(".SoaBEf"):
        if articles_fetched >= FETCH_LIMIT:
            break

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

        articles_fetched += 1

        if len(valid_articles) >= MAX_VALID_ARTICLES:
            break

    return valid_articles


def save_articles(stock, articles):
    saved_files = []

    for i, article in enumerate(articles):
        filename = f"{article['date']}_{stock}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Handle duplicate file names (appending _1, _2, etc.)
        counter = 1
        while os.path.exists(filepath):
            filepath = os.path.join(OUTPUT_DIR, f"{article['date']}_{stock}_{counter}.txt")
            counter += 1

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Stock: {stock}\n")
            f.write(f"Date: {article['date']}\n")
            f.write(f"Title: {article['title']}\n")
            f.write(f"URL: {article['url']}\n")
            f.write(f"Full Article:\n{article['full_article']}\n")

        saved_files.append(filepath)
        print(f"Saved: {filepath}")

    return len(saved_files) > 0


def fetch_recent_ohlc(stock):
    try:
        ticker = yf.Ticker(stock)
        df = ticker.history(period="1d")

        if not df.empty:
            data = {
                "Stock": stock,
                "Open_Price": df.iloc[-1]["Open"],
                "Close_Price": df.iloc[-1]["Close"],
                "High_Price": df.iloc[-1]["High"],
                "Low_Price": df.iloc[-1]["Low"],
                "Volume": df.iloc[-1]["Volume"],
            }
            print(f"OHLC Data Fetched: {stock} - Close: {data['Close_Price']}")
            return data
        else:
            print(f"No OHLC data found for {stock}")
            return None

    except Exception as e:
        print(f"Failed to fetch OHLC data for {stock}: {e}")
        return None


def fetch_past_21_days_ohlc(stock):
    try:
        ticker = yf.Ticker(stock)
        df = ticker.history(period="21d")

        if not df.empty:
            df["Stock"] = stock
            df.reset_index(inplace=True)
            df["Date"] = df["Date"].dt.tz_localize(None)  # Remove timezone
            df = df[["Date", "Stock", "Open", "High", "Close", "Low", "Volume"]]
            return df
        else:
            print(f"No past 21 days OHLC data found for {stock}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Failed to fetch past 21 days OHLC data for {stock}: {e}")
        return pd.DataFrame()


all_ohlc_data = []
past_21_days_ohlc_data = []

for stock in stocks:
    # 1. Scrape and Save News Articles
    articles = scrape_news(stock)
    saved_articles = save_articles(stock, articles) if articles else False

    # 2. Fetch OHLC Data only if news was saved
    if saved_articles:
        ohlc_data = fetch_recent_ohlc(stock)
        if ohlc_data:
            all_ohlc_data.append(ohlc_data)

        # 3. Fetch past 21 days OHLC Data
        past_21_days_df = fetch_past_21_days_ohlc(stock)
        if not past_21_days_df.empty:
            past_21_days_ohlc_data.append(past_21_days_df)

    # Pause between requests
    time.sleep(random.uniform(5, 10))

# Save OHLC Data to Excel
if all_ohlc_data:
    ohlc_df = pd.DataFrame(all_ohlc_data)
    ohlc_df.to_excel("stock_ohlc_data.xlsx", index=False)
    print("\nOHLC data saved to 'stock_ohlc_data_ex.xlsx'")
else:
    print("\nNo OHLC data to save.")

# Save Past 21 Days OHLC Data to Excel
if past_21_days_ohlc_data:
    past_21_days_df = pd.concat(past_21_days_ohlc_data, ignore_index=True)
    past_21_days_df.to_excel("stock_ohlc_past_21_days.xlsx", index=False)
    print("\nPast 21 days OHLC data saved to 'Stock_ohlc_past_21_days.xlsx'")
else:
    print("\nNo past 21 days OHLC data to save.")

print("\nNews scraping and OHLC data fetching completed!")