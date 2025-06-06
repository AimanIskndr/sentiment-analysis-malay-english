import pandas as pd
import re

def clean_text(text):
    # Remove mentions, hashtags, and links
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove links
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def clean_dataset(file_path, output_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Clean the text column
    df['text'] = df['text'].apply(clean_text)
    
    # Save the cleaned dataset
    df.to_csv(output_path, index=False)