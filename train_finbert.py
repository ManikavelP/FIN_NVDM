import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from finbert_model import FinBERTClassifier
import json

def train_finbert():
    print("--- Training FinBERT on StockEmotions ---")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
        
    stock_dir = os.path.join(base_dir, "kaggle_stockEmotions_dataset", "tweet")
    
    # 1. Load Training Data (Resampled)
    print("Loading Training Data...")
    train_ids = np.load(os.path.join(stock_dir, "stockemo_finbert_train_ids_resampled.npy"))
    train_mask = np.load(os.path.join(stock_dir, "stockemo_finbert_train_mask_resampled.npy"))
    train_labels_raw = np.load(os.path.join(stock_dir, "stockemo_finbert_train_ids_labels_resampled.npy"), allow_pickle=True)
    
    # Encode Labels
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels_raw)
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {le.classes_}")
    
    # Save Label Encoder Mapping
    label_map = {int(i): label for i, label in enumerate(le.classes_)}
    with open(os.path.join(stock_dir, "stockemo_label_map.json"), "w") as f:
        json.dump(label_map, f)
        
    # Convert to Tensors
    train_dataset = TensorDataset(
        torch.tensor(train_ids, dtype=torch.long),
        torch.tensor(train_mask, dtype=torch.long),
        torch.tensor(train_labels, dtype=torch.long)
    )
    
    # 2. Load Validation Data (Original)
    print("Loading Validation Data...")
    val_ids = np.load(os.path.join(stock_dir, "stockemo_finbert_val_ids.npy"))
    val_mask = np.load(os.path.join(stock_dir, "stockemo_finbert_val_mask.npy"))
    
    # Load Validation Labels from CSV (as they weren't saved as NPY)
    # Note: prepare_data.py read 'val_stockemo_preprocessed.csv' for IDs.
    val_df = pd.read_csv(os.path.join(stock_dir, "val_stockemo_preprocessed.csv")).dropna(subset=['processed_content'])
    val_labels_raw = val_df['emo_label'].values
    
    # Transform Val Labels
    # Handle unseen labels? StockEmo shouldn't have unseen labels in Val usually.
    val_labels = le.transform(val_labels_raw)
    
    val_dataset = TensorDataset(
        torch.tensor(val_ids, dtype=torch.long),
        torch.tensor(val_mask, dtype=torch.long),
        torch.tensor(val_labels, dtype=torch.long)
    )
    
    # DataLoaders
    BATCH_SIZE = 16 # BERT is heavy
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 3. Model Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")
    
    model = FinBERTClassifier(num_classes=num_classes)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    # Fine-tuning typically uses lower LR
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # 4. Training Loop
    EPOCHS = 3
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (ids, mask, y) in enumerate(train_loader):
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Acc
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1} [{i}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {correct/total:.4f}")
        
        train_acc = correct / total
        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f} Train Acc: {train_acc:.4f}")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for ids, mask, y in val_loader:
                ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
                logits = model(ids, mask)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} Val Acc: {val_acc:.4f}")
        
        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(stock_dir, "finbert_stockemo_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New Best Model Saved! (Acc: {best_acc:.4f})")
            
    print("Training Complete.")

if __name__ == "__main__":
    train_finbert()
