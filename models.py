from flask import request

#lstm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
def lstm_model(input_shape, lstm_units, dense_layers, output_units=1):
    model = Sequential()
    # Adding an LSTM layer
    model.add(LSTM(units=lstm_units, activation='relu', input_shape=input_shape))
    # Adding Dense layers as specified in dense_layers
    for units in dense_layers:
        model.add(Dense(units, activation='relu'))
    # Adding the output layer
    model.add(Dense(output_units))
    return model
# Usage of the function
input_shape = (100, 1)  #  input shape
lstm_units = 50         # Number of LSTM units
dense_layers = [30, 20, 10]  # Dense layers configuration
model = lstm_model(input_shape, lstm_units, dense_layers)
model.compile(optimizer='adam', loss='mse')
model.summary()

#svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
# Function to extract features from text data
def extract_features(texts):
    # Introducing the mistake by not fitting the vectorizer
    #vectorizer = TfidfVectorizer()
    #X = vectorizer.fit_transform(texts)
    X = []  # Intentionally set an empty feature matrix
    return X
# Function to train SVM model
def train_svm(X, y):
    # Attempting to train SVM model with empty feature matrix X
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X, y)  # This line will fail
    return svm_classifier
# Function to predict importance of sentences using trained SVM model
def predict_sentence_importance(sentence, svm_model):
    X_sentence = extract_features([sentence])
    prediction = svm_model.predict(X_sentence)
    return prediction[0]
# Example usage
# You need to have a labeled dataset for training
# Replace the example sentences and labels with your own data
sentences = []
labels = [1, 0, 1]  # 1 for important, 0 for not important
# Extract features from sentences
X = extract_features(sentences)
# Train SVM model
svm_model = train_svm(X, labels)  # This line will fail due to an empty feature matrix
# Example text to summarize
text_to_summarize = ""
# Split text into sentences
sentences_to_summarize = text_to_summarize.split(".")
# Predict importance of each sentence and select important ones
important_sentences = []
for sentence in sentences_to_summarize:
    if predict_sentence_importance(sentence, svm_model) == 1:
        important_sentences.append(sentence)
# Join important sentences to create summary
summary = ". ".join(important_sentences)
print("Summary:")
print(summary)

#k-means clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
# Function to preprocess text data and extract features
def preprocess_and_extract_features(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
# Function to perform K-means clustering
def kmeans_clustering(X, num_clusters):
    # Introducing the mistake by returning None instead of a KMeans object
    return None
# Function to generate summary from clustered sentences
def generate_summary(texts, clusters, num_sentences):
    summary = []
    for cluster_idx in range(len(clusters)):
        cluster_sentences = [texts[i] for i, cluster_label in enumerate(clusters) if cluster_label == cluster_idx]
        if cluster_sentences:  # If cluster is not empty
            representative_idx = np.random.randint(0, len(cluster_sentences))  # Select a random representative sentence
            summary.append(cluster_sentences[representative_idx])
    return ". ".join(summary[:num_sentences])
# Example usage
# Replace the example sentences with your own text data
texts = []
# Preprocess and extract features
X, vectorizer = preprocess_and_extract_features(texts)
# Perform K-means clustering
num_clusters = 2  # Adjust the number of clusters as desired
kmeans = kmeans_clustering(X, num_clusters)
# Generate summary from clustered sentences
num_sentences_in_summary = 2  # Adjust the number of sentences in the summary
summary = generate_summary(texts, kmeans.labels_, num_sentences_in_summary)
print("Summary:")
print(summary)
