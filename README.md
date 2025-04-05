📰 Fake News Detector with Machine Learning

This project is a simple but effective machine learning model that helps detect fake news articles. It uses a dataset of real and fake news headlines and body texts to train a model that can predict whether a given article is real or fake.





🔍 What's Inside?

TF-IDF Vectorizer: Converts text data into numbers that the model can understand.

PassiveAggressiveClassifier: A fast and lightweight model used for text classification.

Accuracy Score & Confusion Matrix: Helps evaluate how well the model performs.


📁 Files You'll Need

Make sure you have the following CSV files in your working directory:

True.csv — contains real news articles

Fake.csv — contains fake news articles


💡 Each CSV should have a column named text which includes the news article content.


🛠️ How It Works

The data is loaded and labeled (1 for real, 0 for fake).

The text data is split into training and testing sets.

A TF-IDF vectorizer is used to convert text into feature vectors.

A PassiveAggressiveClassifier is trained on the vectors.

The model is tested and the accuracy is printed.

A confusion matrix is shown to visualize the performance.


📊 Results

On the sample dataset, this model achieves an accuracy of around 90-95% depending on your data split and environment.


▶️ How to Run

This code is designed to run in Google Colab:

Open the Colab notebook or upload the .py file.

Upload True.csv and Fake.csv when prompted.

Run all the cells to train and test the model.


💬 Output Example

lua
Copy
Edit
Data split successful!
Accuracy: 93.82%
[[589   49]
 [  9 619]]


🧠 Technologies Used

Python 🐍

Pandas

NumPy

Scikit-learn

Google Colab


🙌 Credits
Originally adapted from a Google Colab project. This is a lightweight example to help understand text classification basics using machine learning.

