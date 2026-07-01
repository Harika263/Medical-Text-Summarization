from flask import request
import numpy as np
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
# Usage of the function
input_shape = (100, 1)
lstm_units = 50
dense_layers = [30, 20, 10]

model = lstm_model(input_shape, lstm_units, dense_layers)

model.compile(
    optimizer='adam',
    loss='mse'
)

# Dummy training data
X_train = np.random.rand(100, 100, 1)
y_train = np.random.rand(100)

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=5
)

model.summary()

#svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
# Function to extract features from text data
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
# Function to train SVM model
def train_svm(X, y):
    # Attempting to train SVM model with empty feature matrix X
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X, y)  # This line will fail
    return svm_classifier
# Function to predict importance of sentences using trained SVM model
# Function to predict importance of sentences using trained SVM model
def predict_sentence_importance(sentence, svm_model):
    X_sentence, _ = extract_features([sentence])
    prediction = svm_model.predict(X_sentence)
    return prediction[0]
# Example usage
# You need to have a labeled dataset for training
# Replace the example sentences and labels with your own data
# Example training data
sentences = [
    "This sentence is important",
    "This is not important",
    "Another important sentence"
]

labels = [1, 0, 1]

# Extract features
X, vectorizer = extract_features(sentences)

# Train SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, labels)

print("SVM model trained successfully")
# Example text to summarize
text_to_summarize = """
Machine learning is transforming healthcare.
Text summarization is an NLP task.
Artificial intelligence is evolving rapidly.
"""
# Split text into sentences
sentences_to_summarize = text_to_summarize.split(".")
# Predict importance of each sentence and select important ones
important_sentences = []
for sentence in sentences_to_summarize:
    if predict_sentence_importance(sentence, svm_classifier) == 1:
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
from sklearn.cluster import KMeans

def kmeans_clustering(X, num_clusters):
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42
    )
    kmeans.fit(X)
    return kmeans
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
# Example usage
texts = [
    "Machine learning is useful",
    "Artificial intelligence is growing rapidly",
    "Text summarization is an NLP task",
    "Deep learning models require large datasets"
]

# Preprocess and extract features
X, vectorizer = preprocess_and_extract_features(texts)

# Perform K-means clustering
num_clusters = 2
kmeans = kmeans_clustering(X, num_clusters)

# Generate summary
num_sentences_in_summary = 2
summary = generate_summary(
    texts,
    kmeans.labels_,
    num_sentences_in_summary
)

print("Summary:")
print(summary)
