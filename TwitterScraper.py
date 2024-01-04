#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from ntscraper import Nitter
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.models import load_model
import os
from sklearn.preprocessing import LabelEncoder
plt.style.use('ggplot')

max_sequence_length = 280
tokenizer = Tokenizer(num_words=1000)

# Read in data
column_names = ['target', 'id', 'date', 'query', 'user', 'text']
df = pd.read_csv('text_dataset.csv', names=column_names, encoding='cp1252')

df['text'] = df['text'].str.lower()

# Prepare tokenizer
tokenizer.fit_on_texts(df['text'])

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(df['text'])

# Padding sequences to ensure uniform length
data = pad_sequences(sequences, maxlen=max_sequence_length)

df['target'] = df['target'].replace(4, 1)

target = df['target'].values


if os.path.exists('my_model.h5'):
    model = load_model('my_model.h5')
else:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Display the shapes of the splits
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Neural network architecture
    model = Sequential()
    model.add(Embedding(input_dim=1000, output_dim=50, input_length=max_sequence_length))  # Reduced output_dim
    model.add(LSTM(50))  # Reduced number of units
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with a potentially larger learning rate
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train with a larger batch size and potentially fewer epochs
    model.fit(X_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

    # Save the model
    model.save('my_model.h5')



now = datetime.now()
yesterday = now - timedelta(days=1)

now_str = now.strftime("%Y-%m-%d")
yesterday_str = yesterday.strftime("%Y-%m-%d")

tweet_text = []

username = input("Enter your a Twitter user name: ")


scraper = Nitter()

tweets_data = scraper.get_tweets(username, mode='user', since=yesterday_str, until=now_str)

if 'tweets' in tweets_data:
    for tweet in tweets_data['tweets']:
        tweet_text.append(tweet['text'])
else:
    print("No 'tweets' key found in the data")
    
tweet_sequences = tokenizer.texts_to_sequences(tweet_text)
tweet_data = pad_sequences(tweet_sequences, maxlen=max_sequence_length)


predictions = model.predict(tweet_data)

total_score = 0

print(f"User: {username}")
for tweet, sentiment in zip(tweet_text, predictions):
    if sentiment > 0.5:
        sentiment_label = 'Positive'
        total_score += 1
    else:
        sentiment_label = 'Negative'
    print(f"Tweet: {tweet}\nSentiment: {sentiment_label}\n")

total_score = (total_score / len(tweet_data)) * 100
print(f"Overall User: {username} was {total_score}% positive today!")

    