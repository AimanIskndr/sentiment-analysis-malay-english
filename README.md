# Sentiment Analysis of Malay and English Tweets

This project aims to perform sentiment analysis on a dataset of Malay and English tweets using various machine learning models. The project includes data cleaning, implementation of Naive Bayes and other traditional sentiment analysis algorithms, as well as embedding-based models. The effectiveness of these models will be compared using confusion matrices and other performance metrics.

## Project Structure

```
sentiment-analysis-malay-english
├── data
│   ├── raw
│   │   └── semisupervised-bert-xlnet.csv  # Raw dataset of tweets
│   ├── processed                            # Directory for cleaned and processed data
│   └── splits                               # Directory for training and testing splits
├── src
│   ├── data_preprocessing                   # Module for data preprocessing
│   │   ├── __init__.py
│   │   ├── data_cleaner.py                  # Functions to clean the dataset
│   │   └── text_preprocessor.py              # Functions for text preprocessing tasks
│   ├── models                               # Module for sentiment analysis models
│   │   ├── __init__.py
│   │   ├── naive_bayes.py                   # Implementation of Naive Bayes algorithm
│   │   ├── embedding_models.py               # Embedding-based models
│   │   └── traditional_models.py             # Traditional sentiment analysis algorithms
│   ├── evaluation                            # Module for model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                        # Functions for performance metrics
│   │   └── confusion_matrix.py               # Functions for confusion matrix visualization
│   ├── utils                                 # Utility functions
│   │   ├── __init__.py
│   │   └── data_split.py                     # Functions for dataset splitting
│   └── main.py                              # Entry point for the project
├── notebooks                                 # Jupyter notebooks for exploration and analysis
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── results                                   # Directory for storing results
│   ├── confusion_matrices                   # Confusion matrices for each model
│   ├── performance_metrics                   # Performance metrics results
│   └── model_comparison                      # Comparison results of different models
├── requirements.txt                          # Project dependencies
├── config.yaml                               # Configuration settings
└── README.md                                 # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd sentiment-analysis-malay-english
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   Place the raw dataset `semisupervised-bert-xlnet.csv` in the `data/raw` directory.

4. **Run the Project**:
   Execute the main script to start the sentiment analysis process:
   ```
   python src/main.py
   ```

## Objectives

- To clean and preprocess a dataset of Malay and English tweets.
- To implement and compare various sentiment analysis models.
- To evaluate model performance using confusion matrices and other metrics.

## Acknowledgments

This project utilizes various libraries and tools for data processing and machine learning. Special thanks to the contributors of the datasets and libraries used.