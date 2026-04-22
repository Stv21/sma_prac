import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/content/drive/MyDrive/Tweets.csv")

print("Basic Engagement Stats:")
print(df["retweet_count"].describe())


eng_sentiment = df.groupby("airline_sentiment")["retweet_count"].mean()

print("\nAvg Engagement by Sentiment:")
print(eng_sentiment)

plt.figure()
eng_sentiment.plot(kind='bar')
plt.title("Engagement by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Average Retweets")
plt.show()


eng_airline = df.groupby("airline")["retweet_count"].mean()

print("\nAvg Engagement by Airline:")
print(eng_airline)

plt.figure()
eng_airline.plot(kind='bar')
plt.title("Engagement by Airline")
plt.xlabel("Airline")
plt.ylabel("Average Retweets")
plt.xticks(rotation=45)
plt.show()


df["tweet_length"] = df["text"].astype(str).apply(len)

plt.figure()
plt.scatter(df["tweet_length"], df["retweet_count"])
plt.title("Tweet Length vs Engagement")
plt.xlabel("Tweet Length")
plt.ylabel("Retweet Count")
plt.show()


top_tweets = df.sort_values(by="retweet_count", ascending=False).head(5)

print("\nTop Engaging Tweets:")
print(top_tweets[["text", "retweet_count"]])