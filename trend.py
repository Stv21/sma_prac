import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/content/drive/MyDrive/Tweets.csv")


df["tweet_created"] = pd.to_datetime(df["tweet_created"], errors='coerce')

df["date"] = df["tweet_created"].dt.date
daily_trend = df["date"].value_counts().sort_index()

print("\nDaily Tweet Trend:")
print(daily_trend)


plt.figure()
daily_trend.plot()
plt.title("Tweet Trend Over Time (Daily)")
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.show()


df["hour"] = df["tweet_created"].dt.hour
hourly_trend = df["hour"].value_counts().sort_index()

print("\nHourly Tweet Trend:")
print(hourly_trend)


plt.figure()
hourly_trend.plot(kind='bar')
plt.title("Tweet Activity by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Tweets")
plt.show()


sentiment_trend = df.groupby(["date", "airline_sentiment"]).size().unstack()

print("\nSentiment Trend:")
print(sentiment_trend)


plt.figure()
sentiment_trend.plot()
plt.title("Sentiment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.show()