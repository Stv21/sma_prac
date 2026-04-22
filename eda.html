import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/drive/MyDrive/Datasets/Tweets.csv")


print("Dataset Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


sentiment_counts = df["airline_sentiment"].value_counts()

plt.figure()
sentiment_counts.plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()


airline_counts = df["airline"].value_counts()

plt.figure()
airline_counts.plot(kind='bar')
plt.title("Tweets per Airline")
plt.xlabel("Airline")
plt.ylabel("Count")
plt.show()


sentiment_airline = pd.crosstab(df["airline"], df["airline_sentiment"])

plt.figure()
sentiment_airline.plot(kind='bar', stacked=True)
plt.title("Sentiment by Airline")
plt.xlabel("Airline")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

neg_reason = df["negativereason"].value_counts().head(10)

plt.figure()
neg_reason.plot(kind='bar')
plt.title("Top Negative Reasons")
plt.xlabel("Reason")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


df["tweet_created"] = pd.to_datetime(df["tweet_created"], errors='coerce')
df["date"] = df["tweet_created"].dt.date

daily_trend = df["date"].value_counts().sort_index()

plt.figure()
daily_trend.plot()
plt.title("Tweets Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.show()


df["tweet_length"] = df["text"].astype(str).apply(len)

plt.figure()
df["tweet_length"].plot(kind='hist')
plt.title("Tweet Length Distribution")
plt.xlabel("Length")
plt.show()