import pandas as pd
from src.data_preprocessing.data_cleaner import clean_data
from src.data_preprocessing.text_preprocessor import preprocess_text
from src.utils.data_split import split_data
from src.models.naive_bayes import train_naive_bayes
from src.models.embedding_models import train_embedding_models
from src.models.traditional_models import train_traditional_models
from src.evaluation.metrics import calculate_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix

def main():
    # Load the dataset
    data_path = 'data/raw/semisupervised-bert-xlnet.csv'
    df = pd.read_csv(data_path)

    # Data cleaning
    cleaned_data = clean_data(df)

    # Text preprocessing
    preprocessed_data = preprocess_text(cleaned_data)

    # Split the data into training and testing sets (80-20 split)
    train_data, test_data = split_data(preprocessed_data, test_size=0.2)

    # Train models
    naive_bayes_model = train_naive_bayes(train_data)
    embedding_model = train_embedding_models(train_data)
    traditional_model = train_traditional_models(train_data)

    # Evaluate models
    naive_bayes_metrics = calculate_metrics(naive_bayes_model, test_data)
    embedding_metrics = calculate_metrics(embedding_model, test_data)
    traditional_metrics = calculate_metrics(traditional_model, test_data)

    # Plot confusion matrices
    plot_confusion_matrix(naive_bayes_model, test_data)
    plot_confusion_matrix(embedding_model, test_data)
    plot_confusion_matrix(traditional_model, test_data)

    # Print metrics
    print("Naive Bayes Metrics:", naive_bayes_metrics)
    print("Embedding Model Metrics:", embedding_metrics)
    print("Traditional Model Metrics:", traditional_metrics)

if __name__ == "__main__":
    main()