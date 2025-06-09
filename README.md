# Sentiment Analysis of Malay and English Tweets

This project aims to perform sentiment analysis on a dataset of Malay and English tweets using various machine learning models. The project includes data cleaning, implementation of Naive Bayes and other traditional sentiment analysis algorithms, as well as embedding-based models. The effectiveness of these models will be compared using confusion matrices and other performance metrics.

## Project Structure

```
sentiment-analysis-malay-english/
├── config.yaml
├── README.md
├── requirements.txt
├── data/
│   ├── splits/
│   ├── processed/
│   │   └── cleaned_data.csv
│   └── raw/
│       └── semisupervised-bert-xlnet.csv
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   └── model_train_and_eval.ipynb
├── results/
│   ├── confusion_matrices/
│   ├── model_comparison/
│   └── performance_metrics/
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AimanIskndr/sentiment-analysis-malay-english
   cd sentiment-analysis-malay-english
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   Place the raw dataset `semisupervised-bert-xlnet.csv` in the `data/raw` directory.

4. **Run the Project**:
   Execute the main script to start the sentiment analysis process:
   ```bash
   python src/main.py
   ```

## Objectives

- To clean and preprocess a dataset of Malay and English tweets.
- To implement and compare various sentiment analysis models.
- To evaluate model performance using confusion matrices and other metrics.

## Models Implemented

1. **NLTK Sentiment Analysis (VADER)**: Rule-based sentiment analysis.
2. **Multinomial Naive Bayes (MNB)**: Using TF-IDF features.
3. **Linear Support Vector Machine (SVM)**: Using TF-IDF features.
4. **Logistic Regression**: Using sentence embeddings from `paraphrase-multilingual-MiniLM-L12-v2`.

## Evaluation Metrics

- **Accuracy**
- **Macro-averaged F1 Score**
- **Confusion Matrices**

## Results

The following models were evaluated:

| Model                  | Accuracy | Macro-F1 |
|------------------------|----------|----------|
| **VADER**              | 0.XXX    | 0.XXX    |
| **Linear SVM (TF-IDF)**| 0.XXX    | 0.XXX    |
| **Multinomial NB (TF-IDF)** | 0.XXX | 0.XXX    |
| **MiniLM + Logistic Regression** | 0.XXX | 0.XXX |

(Note: Replace `0.XXX` with actual results after running the evaluation.)

## Acknowledgments

This project utilizes various libraries and tools for data processing and machine learning. Special thanks to the contributors of the datasets and libraries used, including:

- Pre-trained FastText embeddings: [FastText](https://fasttext.cc/)
- Sentence embeddings: [SentenceTransformers](https://www.sbert.net/)
- NLTK VADER Sentiment Analysis: [NLTK](https://www.nltk.org/)