import torch
import os
import sys

def verify():
    print("Verifying FinBERT Scripts Syntax...")
    
    # Check imports
    try:
        from finbert_model import FinBERTClassifier
        print("finbert_model imported successfully.")
        
        import train_finbert
        print("train_finbert imported successfully.")
        
        import evaluate_finbert
        print("evaluate_finbert imported successfully.")
        
    except ImportError as e:
        print(f"Import Error: {e}")
        return

    # Check Model Instantiation
    try:
        model = FinBERTClassifier(num_classes=12)
        print("FinBERTClassifier instantiated successfully.")
        print(model)
    except Exception as e:
        print(f"Model Instantiation Error: {e}")

if __name__ == "__main__":
    verify()
