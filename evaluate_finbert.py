import torch
import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from finbert_model import FinBERTClassifier

def evaluate_finbert():
    print("--- Evaluating FinBERT on StockEmotions (Test Set) ---")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
        
    stock_dir = os.path.join(base_dir, "kaggle_stockEmotions_dataset", "tweet")
    
    # 1. Load Test Data
    print("Loading Test Data...")
    test_ids = np.load(os.path.join(stock_dir, "stockemo_finbert_test_ids.npy"))
    test_mask = np.load(os.path.join(stock_dir, "stockemo_finbert_test_mask.npy"))
    
    # Load Test Labels from CSV
    test_df = pd.read_csv(os.path.join(stock_dir, "test_stockemo_preprocessed.csv")).dropna(subset=['processed_content'])
    test_labels_raw = test_df['emo_label'].values
    
    # Load Label Map
    with open(os.path.join(stock_dir, "stockemo_label_map.json"), "r") as f:
        label_map = json.load(f)
        # Convert keys to int (json keys are strings)
        label_map = {int(k): v for k, v in label_map.items()}
    
    # Invert map for encoding
    label2id = {v: k for k, v in label_map.items()}
    
    # Encode Test Labels
    try:
        test_labels = [label2id[l] for l in test_labels_raw]
        test_labels = np.array(test_labels)
    except KeyError as e:
        print(f"Error encoding test labels: {e}. Check if Test classes match Train classes.")
        return

    # 2. Load Model
    num_classes = len(label_map)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")
    model = FinBERTClassifier(num_classes=num_classes)
    
    model_path = os.path.join(stock_dir, "finbert_stockemo_best.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 3. Inference
    batch_size = 32
    all_preds = []
    
    test_tensor_ids = torch.tensor(test_ids, dtype=torch.long)
    test_tensor_mask = torch.tensor(test_mask, dtype=torch.long)
    
    print("Running Inference...")
    with torch.no_grad():
        for i in range(0, len(test_ids), batch_size):
            b_ids = test_tensor_ids[i:i+batch_size].to(DEVICE)
            b_mask = test_tensor_mask[i:i+batch_size].to(DEVICE)
            
            logits = model(b_ids, b_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            
    # 4. Metrics
    acc = accuracy_score(test_labels, all_preds)
    f1 = f1_score(test_labels, all_preds, average='weighted')
    prec = precision_score(test_labels, all_preds, average='weighted')
    rec = recall_score(test_labels, all_preds, average='weighted')
    
    print(f"\nResults:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print(f"Precision (Weighted): {prec:.4f}")
    print(f"Recall (Weighted): {rec:.4f}")
    
    print("\nClassification Report:")
    target_names = [label_map[i] for i in range(num_classes)]
    print(classification_report(test_labels, all_preds, target_names=target_names))

if __name__ == "__main__":
    evaluate_finbert()
