import os
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

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