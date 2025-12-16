import numpy as np
import pandas as pd
import os
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
import scipy.sparse
from scipy.sparse import csr_matrix, save_npz

def balance_dataset(name, X, y, method='ros', strategy='auto'):
    print(f"Balancing {name} using {method}...")
    print(f"Original shape: {X.shape}")
    print(f"Original Class Distribution: {Counter(y)}")
    
    # Ensure X is sparse if it's BoW (2D and large vocab)
    is_sparse = False
    if X.ndim == 2 and X.shape[1] > 1000: # Heuristic for BoW
        if not scipy.sparse.issparse(X):
             print("Converting to sparse CSR matrix to save memory...")
             X = csr_matrix(X)
        is_sparse = True
    
    sampler = None
    if method == 'ros':
        sampler = RandomOverSampler(sampling_strategy=strategy, random_state=42)
    elif method == 'smote':
        min_samples = pd.Series(y).value_counts().min()
        if min_samples < 2:
            print("Warning: Contains classes with < 2 samples. Fallback to ROS.")
            sampler = RandomOverSampler(sampling_strategy=strategy, random_state=42)
        else:
            k = min(5, min_samples - 1)
            # Support sparse input for SMOTE
            sampler = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=42)
            
    try:
        X_res, y_res = sampler.fit_resample(X, y)
        print(f"Resampled shape: {X_res.shape}")
        return X_res, y_res
    except Exception as e:
        print(f"Error during resampling: {e}. Fallback to ROS.")
        ros = RandomOverSampler(sampling_strategy=strategy, random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        print(f"Resampled shape (ROS fallback): {X_res.shape}")
        return X_res, y_res

def save_resampled(output_dir, prefix, X, y):
    if scipy.sparse.issparse(X):
        save_npz(os.path.join(output_dir, f"{prefix}_resampled.npz"), X)
    else:
        np.save(os.path.join(output_dir, f"{prefix}_resampled.npy"), X)
    np.save(os.path.join(output_dir, f"{prefix}_labels_resampled.npy"), y)

def main():
    base_dir = r"d:\Documents1\Project"
    
    # -------------------------------------------------------------
    # 1. Finsen Dataset
    # -------------------------------------------------------------
    print("\n--- Processing Finsen ---")
    finsen_dir = os.path.join(base_dir, "kaggle_finsen_dataset")
    
    # Load Training Data
    try:
        # Load Dense, convert inside balance_dataset
        print("Loading Finsen BoW...")
        X_bow = np.load(os.path.join(finsen_dir, "finsen_bow_train.npy"))
        ids = np.load(os.path.join(finsen_dir, "finsen_finbert_train_ids.npy"))
        mask = np.load(os.path.join(finsen_dir, "finsen_finbert_train_mask.npy"))
        
        train_df = pd.read_csv(os.path.join(finsen_dir, "train_finsen.csv"))
        y = train_df['Tag'].values
        
        # 1.4.1 NVDM (BoW) -> SMOTE
        # Pass convert_sparse=True logic
        X_bow_res, y_bow_res = balance_dataset("Finsen BoW", X_bow, y, method='smote')
        # Release memory of original X_bow inputs if needed? Local var scope handles it mostly, 
        # but X_bow is still referenced. 
        del X_bow # Hint to GC
        
        save_resampled(finsen_dir, "finsen_bow_train", X_bow_res, y_bow_res)
        
        # 1.4.2 FinBERT -> ROS
        indices = np.arange(len(y)).reshape(-1, 1)
        idx_res, y_res = balance_dataset("Finsen Indices", indices, y, method='ros')
        idx_res = idx_res.flatten()
        
        ids_res = ids[idx_res]
        mask_res = mask[idx_res]
        
        save_resampled(finsen_dir, "finsen_finbert_train_ids", ids_res, y_res)
        np.save(os.path.join(finsen_dir, "finsen_finbert_train_mask_resampled.npy"), mask_res)
        
    except FileNotFoundError as e:
        print(f"Skipping Finsen: {e}")

    # -------------------------------------------------------------
    # 2. StockEmotions Dataset
    # -------------------------------------------------------------
    print("\n--- Processing StockEmotions ---")
    stock_dir = os.path.join(base_dir, "kaggle_stockEmotions_dataset", "tweet")
    
    try:
        print("Loading StockEmo BoW...")
        X_bow = np.load(os.path.join(stock_dir, "stockemo_bow_train.npy"))
        ids = np.load(os.path.join(stock_dir, "stockemo_finbert_train_ids.npy"))
        mask = np.load(os.path.join(stock_dir, "stockemo_finbert_train_mask.npy"))
        
        train_df = pd.read_csv(os.path.join(stock_dir, "train_stockemo_preprocessed.csv")).dropna(subset=['processed_content'])
        y = train_df['emo_label'].values
        
        # 1.4.1 NVDM (BoW) -> SMOTE
        X_bow_res, y_bow_res = balance_dataset("StockEmo BoW", X_bow, y, method='smote')
        del X_bow
        save_resampled(stock_dir, "stockemo_bow_train", X_bow_res, y_bow_res)
        
        # 1.4.2 FinBERT -> ROS
        indices = np.arange(len(y)).reshape(-1, 1)
        idx_res, y_res = balance_dataset("StockEmo Indices", indices, y, method='ros')
        idx_res = idx_res.flatten()
        
        ids_res = ids[idx_res]
        mask_res = mask[idx_res]
        
        save_resampled(stock_dir, "stockemo_finbert_train_ids", ids_res, y_res)
        np.save(os.path.join(stock_dir, "stockemo_finbert_train_mask_resampled.npy"), mask_res)

    except FileNotFoundError as e:
        print(f"Skipping StockEmo: {e}")


if __name__ == "__main__":
    main()
