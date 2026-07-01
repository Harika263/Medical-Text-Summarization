from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import regex
from flask import Flask, render_template, request
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from summa.summarizer import summarize as summa_summarize # Importing from summa
from collections import defaultdict
from collections.abc import Mapping

# Removed unnecessary print statement for sentiment analysis


def read_article(data):
    article = data.split(". ")
    sentences = []
    for sentence in article:
        review = regex.sub("[^A-Za-z0-9]", ' ', sentence)
        sentences.append(review.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)  # makes a vector of len all_words
    vector2 = [0] * len(all_words)
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    # Autoencoders
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - nltk.cluster.util.cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def generate_summary(file_name, top_n=10):
    stop_words = stopwords.words('english')
    summarize_text = []
    # Step 1 - Read text and split it
    sentences = read_article(file_name)
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    #     print("\n\n---------------\nIndexes of top ranked_sentence order are ", ranked_sentence)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    # Step 5 - Offcourse, output the summarize texr
    # print("\n")
    # print("*"*140)
    # print("\n\nSUMMARY: \n---------\n\n", ". ".join(summarize_text))
    a = ". ".join(summarize_text)
    return a

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"
import requests

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/summary',methods=["GET","POST"])
def summary():
    if request.method == "POST":
    # # f1=int(request.form['city'])
            f2=request.form['input_text']
            # f3 = request.form['num_sentences']
            print(f2)
            # print(f3)
            print(type(f2))
            summary = summa_summarize(f2, ratio =0.45)
            return render_template('summary.html', title="Summarizer", original_text=f2, output_summary=summary,
                           num_sentences=5)
    return render_template('summary.html')

@app.route('/keyword', methods=["GET", "POST"])
def keyword():
    if request.method == "POST":
        f4 = request.form['input_text']
        print(type(f4))
        # Tokenize the text
        words = word_tokenize(f4)
        # Filter out stopwords and non-alphabetic words
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalpha() and word.lower() not in stop_words]
        # Calculate frequency distribution
        fdist = FreqDist(filtered_words)
        # Get the most common words
        common_words = [word for word, frequency in fdist.most_common(10)]
        return render_template('keyword.html', title="Key Word Extractor", original_text=f4, output_summary=", ".join(common_words), num_keywords=5)
    return render_template('keyword.html')


@app.route('/abstract', methods=["GET", "POST"])
def abstract():
    if request.method == "POST":
        f6 = request.form['input_text']
        print(type(f6))
        # Use Summa's summarize function
        summary_text = summa_summarize(f6, ratio=0.45)  # Adjust the ratio as needed
        return render_template('abstract.html', title="Abstractive Summarization", original_text=f6, output_summary=summary_text)
    return render_template('abstract.html')

if __name__=="__main__":
    app.run(debug=True,port=8000)
