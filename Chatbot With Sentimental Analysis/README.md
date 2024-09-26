# Simple Sentiment Analysis Using Logistic Regression

This repository contains a simple sentiment analysis project using Logistic Regression to classify tweets as positive or negative. The model is trained on the Sentiment140 dataset from Kaggle.

## Project Overview

The goal of this project is to classify the sentiment of tweets into two categories: **positive** or **negative**. We use the Sentiment140 dataset, which contains 1.6 million tweets labeled as either positive or negative.

### Dataset

The dataset used for this project is available on Kaggle. You can download the dataset from the following link:
[Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

The dataset contains the following columns:
- `polarity`: The sentiment of the tweet (0 = negative, 4 = positive)
- `id`: The ID of the tweet
- `date`: The date when the tweet was posted
- `query`: Query used for searching (if applicable)
- `user`: The user who tweeted
- `text`: The actual tweet content

### Model Used

We use Logistic Regression as the machine learning algorithm to classify the sentiment of tweets. Logistic Regression is a simple yet effective model for binary classification problems, such as sentiment analysis.

## Project Structure

The repository contains the following files:

- `Simple Sentiment Analysis Using Logistic Regression.ipynb`: The Jupyter notebook containing the code for loading the dataset, preprocessing, training the model, and evaluating the results.
- `README.md`: This file, providing an overview of the project.

## Steps

The steps involved in this project are:

1. **Data Preprocessing**: Cleaning the tweet text by removing special characters, numbers, and punctuations. Converting text to lowercase and tokenizing it.
2. **Feature Extraction**: Converting text data into numerical features using techniques such as **TF-IDF** or **Count Vectorizer**.
3. **Model Training**: Training the Logistic Regression model on the processed text data.
4. **Model Evaluation**: Evaluating the model's performance using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

## Requirements

To run this project, you will need the following Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib
 ```
## Usage

To run the notebook and train the model:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).
2. Open the Jupyter notebook `Simple Sentiment Analysis Using Logistic Regression.ipynb`.
3. Follow the steps in the notebook to preprocess the data, train the model, and evaluate the results.

## Results

The Logistic Regression model provides a reliable classification of tweet sentiments. Below are some of the key results achieved in the project:

- **Accuracy**: `76%`

> The exact metrics may vary based on different feature extraction techniques and hyperparameters used.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to [Kaggle](https://www.kaggle.com/) for providing the dataset.
- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Project by **Thota Rahul**.  
- Connect with me on [LinkedIn](https://www.linkedin.com/in/rahulthota21)

