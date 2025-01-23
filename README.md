# Sentiment Analysis of Tweets

## Project Description

This project focuses on performing sentiment analysis on a dataset of tweets using Python libraries such as TextBlob and scikit-learn. The project involves the following steps:

1. **Data Loading and Preprocessing:** The project starts by loading a dataset of tweets from a CSV file using Pandas. The tweet text is then preprocessed by removing punctuation, converting to lowercase, and removing stop words using NLTK.
2. **Sentiment Analysis with TextBlob:** TextBlob is used to calculate the sentiment polarity of each tweet, which represents the sentiment expressed in the tweet, ranging from -1 (negative) to 1 (positive).
3. **Data Visualization:** The distribution of sentiment polarity scores is visualized using a histogram to understand the overall sentiment of the tweets.
4. **Model Training and Evaluation:** The project utilizes the Naive Bayes algorithm from scikit-learn to build a sentiment classification model. The dataset is split into training and testing sets, and the model is trained on the training set. The model's performance is then evaluated using metrics such as accuracy, classification report, and confusion matrix, providing insights into the model's ability to correctly classify the sentiment of tweets.
5. **Visualization with Seaborn** The Confusion Matrix, which is used to visualize the performance of the model, is visualized using Seaborn to increase readability and provide more insights.

## Project Goal

The goal of this project is to build a model that can accurately predict the sentiment of tweets using machine learning techniques. This can be valuable for understanding public opinion, tracking brand sentiment, and identifying trends on social media.

## Requirements

- Python 3.6 or higher
- Pandas
- NLTK
- TextBlob
- Scikit-learn
- Matplotlib
- Seaborn
2. Install the required libraries:
  ## Usage

1. Upload your dataset (CSV file) to the Colab environment.
2. Run the notebook cells to execute the code.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
