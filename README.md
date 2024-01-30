---

# Twitter Sentiment Analysis

This project utilizes Python to analyze the sentiment of tweets from a specified user within the last day. It leverages libraries such as Pandas, NumPy, Matplotlib, Seaborn, Keras, and a custom Twitter scraping tool to preprocess text data, build a LSTM neural network model, and predict sentiments as positive or negative.

## Getting Started

1. Ensure you have Python 3 installed.
2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn keras scikit-learn
   ```
   Note: Ensure `ntscraper` (a custom module) is available in your environment.

3. Place your tweet dataset as `text_dataset.csv` in the project directory.

4. Run the script:
   ```bash
   python sentiment_analysis.py
   ```

## Features

- Text preprocessing and tokenization with Keras.
- LSTM neural network for sentiment analysis.
- Sentiment prediction for recent tweets of a specified user.
- Visualization of sentiment analysis results.

## Usage

The script prompts for a Twitter username and analyzes tweets from the previous day, classifying each as positive or negative and providing an overall sentiment score.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to discuss potential improvements.

---
