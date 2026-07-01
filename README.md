# Medical-Text-Summarization


## 📌 Overview

The **Medical Text Summarization Web Application** is an NLP-based web application that generates concise summaries from lengthy medical and clinical text documents. It also provides **keyword extraction** and **abstract summarization** features to help users quickly understand medical reports and research documents.

---

## 🚀 Features

* **Text Summarization** using the TextRank algorithm
* **Keyword Extraction** using NLP techniques
* **Abstract Generation** for medical documents
* **Interactive Web Interface** built with Flask

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Framework:** Flask
* **Libraries:** NLTK, NumPy, NetworkX, Regex, Summa, TensorFlow/Keras, Scikit-learn, Transformers

---

## 📂 Project Structure

```text
Medical-Text-Summarization/
│
├── app.py
├── models.py
├── README.md
├── requirements.txt
│
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── summary.html
│   ├── keyword.html
│   └── abstract.html
│
└── static/
```

---

## ⚙️ Installation

```bash
git clone <repository_url>
cd Medical-Text-Summarization

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Download required NLTK resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

---

## 🧠 Methodology

The application follows these steps:

1. Text preprocessing and tokenization
2. Sentence similarity matrix generation
3. Graph construction using NetworkX
4. Sentence ranking using the TextRank algorithm
5. Summary generation and keyword extraction

---

## 📊 Models Explored

* LSTM (Long Short-Term Memory)
* Support Vector Machine (SVM)
* K-Means Clustering

---

## 🎯 Applications

* Medical report summarization
* Clinical document analysis
* Research paper summarization
* Electronic Health Record (EHR) processing

---

## 🔮 Future Enhancements

* Transformer-based summarization (BERT/T5)
* PDF document summarization
* Cloud deployment and Docker integration
* Advanced abstractive summarization models

---

## 👩‍💻 Author

**Harika Kalluru**

Data Science and Machine Learning Enthusiast

---

## 📜 License

This project is intended for educational and research purposes.
