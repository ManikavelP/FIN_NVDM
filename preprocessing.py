import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize(text):
    """1.1.1 Tokenization and Sequence Decomposition"""
    if not isinstance(text, str):
        return []
    return word_tokenize(text)

def normalize_case(tokens):
    """1.1.2 Case Normalization"""
    return [word.lower() for word in tokens]

def remove_stopwords(tokens):
    """1.1.3 Stopword Elimination"""
    return [word for word in tokens if word not in stop_words]

def filter_alphanumeric(tokens):
    """1.1.4 Alphanumeric Filtering"""
    # Keep only tokens that are alphanumeric
    return [word for word in tokens if word.isalnum()]

def lemmatize(tokens):
    """1.1.5 Morphological Standardization (Lemmatization)"""
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_pipeline(text):
    """Executes the full preprocessing pipeline"""
    tokens = tokenize(text)
    tokens = normalize_case(tokens)
    tokens = remove_stopwords(tokens)
    tokens = filter_alphanumeric(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens) # Join back to string for saving, or keep as list if preferred. 
                            # Requirement says "transform... into optimized numerical representations", 
                            # but usually we save the cleaned text first. 
                            # The prompt mentions "ordered list", but typically for CSV storage we join.
                            # I will join them with space for compatibility with standard text processing flow 
                            # unless 'list' is strictly needed for next step. 
                            # Given 1.1.1 output is a list, but usually stored as string in CSV.
                            # I'll stick to joining for the CSV output to be readable.

def process_file(input_path, output_path, column_name):
    print(f"Processing {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return

    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in {input_path}")
        return

    # Apply preprocessing
    df['processed_content'] = df[column_name].apply(preprocess_pipeline)

    try:
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

if __name__ == "__main__":
    # Define paths
    base_dir = r"d:\Documents1\Project"
    
    # Files to process
    tasks = [
        {
            "input": os.path.join(base_dir, "kaggle_finsen_dataset", "FinSen_US_Categorized_Timestamp.csv"),
            "output": os.path.join(base_dir, "kaggle_finsen_dataset", "FinSen_US_Categorized_Timestamp_preprocessed.csv"),
            "col": "Content"
        },
        {
            "input": os.path.join(base_dir, "kaggle_stockEmotions_dataset", "tweet", "train_stockemo.csv"),
            "output": os.path.join(base_dir, "kaggle_stockEmotions_dataset", "tweet", "train_stockemo_preprocessed.csv"),
            "col": "original"
        },
        {
            "input": os.path.join(base_dir, "kaggle_stockEmotions_dataset", "tweet", "val_stockemo.csv"),
            "output": os.path.join(base_dir, "kaggle_stockEmotions_dataset", "tweet", "val_stockemo_preprocessed.csv"),
            "col": "original"
        },
        {
            "input": os.path.join(base_dir, "kaggle_stockEmotions_dataset", "tweet", "test_stockemo.csv"),
            "output": os.path.join(base_dir, "kaggle_stockEmotions_dataset", "tweet", "test_stockemo_preprocessed.csv"),
            "col": "original"
        }
    ]

    for task in tasks:
        process_file(task["input"], task["output"], task["col"])

    print("All processing steps completed.")
