import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer

def stratified_split(df, label_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Performs stratified splitting of the dataframe.
    """
    # Handle rare classes for stratification
    strat_col = df[label_col].copy()
    vc = strat_col.value_counts()
    rare_classes = vc[vc < 3].index # Need at least 3 for 80/10/10 split roughly (actually need 2 for first split? No, need to be in both train and temp). 
                                    # Actually train_test_split needs min 2 in the group to put 1 in each. 
                                    # With 3 splits, it's safer to have more. 
                                    # Let's map anything < 5 to 'Other_Rare' to be safe.
    strat_col[strat_col.isin(rare_classes)] = 'Other_Rare'

    # If 'Other_Rare' has only 1 item (if only 1 rare item total), it will still fail. 
    # Check if 'Other_Rare' count < 2.
    if (strat_col == 'Other_Rare').sum() == 1:
        # Just don't stratify on these at all? 
        # Or just assign them to 'Other_Rare' which hopefully has many.
        # If there are many singletons, 'Other_Rare' will be large.
        pass

    # First split: Train vs Temp (Val + Test)
    try:
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_ratio + test_ratio), 
            stratify=strat_col, 
            random_state=42
        )
    except ValueError:
        # Fallback to random split if stratification fails completely
        print("Warning: Stratified split failed. Falling back to random split.")
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_ratio + test_ratio), 
            random_state=42
        )

    # For second split, we need to stratify temp_df. 
    # We must re-compute rare classes in temp_df or just use the same mapping? 
    # Use mapping from original df? No, indices changed. 
    # Reread strat_col for temp_df.
    strat_col_temp = strat_col.loc[temp_df.index]
    
    # Check for validity in temp split
    vc_temp = strat_col_temp.value_counts()
    if (vc_temp < 2).any():
        print("Warning: Stratified split for Val/Test might fail due to rare classes in Temp. adjusting...")
        # Map rare in temp to 'Other_Rare_Temp' or just fallback
        rare_temp = vc_temp[vc_temp < 2].index
        strat_col_temp[strat_col_temp.isin(rare_temp)] = 'Other_Rare_Temp'

    try:
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=(test_ratio / (val_ratio + test_ratio)), 
            stratify=strat_col_temp, 
            random_state=42
        )
    except ValueError:
        print("Warning: Stratified Val/Test split failed. Falling back to random split.")
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=(test_ratio / (val_ratio + test_ratio)), 
            random_state=42
        )
    
    return train_df, val_df, test_df

def create_bow_matrices(train_texts, val_texts, test_texts):
    """
    Creates Bag-of-Words matrices using CountVectorizer fit on training data.
    """
    vectorizer = CountVectorizer()
    # Fit on train only to prevent leakage
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    
    return X_train, X_val, X_test, vectorizer

def create_finbert_inputs(texts, tokenizer, max_len=128):
    """
    Tokenizes texts for FinBERT using the tokenizer.
    Returns input_ids and attention_masks.
    """
    encoded = tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='np'
    )
    return encoded['input_ids'], encoded['attention_mask']

def save_numpy_arrays(output_dir, prefix, data_dict):
    """Helper to save numpy arrays."""
    for key, value in data_dict.items():
        np.save(os.path.join(output_dir, f"{prefix}_{key}.npy"), value)

def main():
    base_dir = r"d:\Documents1\Project"
    finsen_dir = os.path.join(base_dir, "kaggle_finsen_dataset")
    stockemo_dir = os.path.join(base_dir, "kaggle_stockEmotions_dataset", "tweet")
    
    # ---------------------------------------------------------
    # 1. Processing Finsen Dataset
    # ---------------------------------------------------------
    print("Processing Finsen Dataset...")
    finsen_path = os.path.join(finsen_dir, "FinSen_US_Categorized_Timestamp_preprocessed.csv")
    if os.path.exists(finsen_path):
        df_finsen = pd.read_csv(finsen_path)
        # Drop rows with NaN in processed_content (if any)
        df_finsen = df_finsen.dropna(subset=['processed_content'])
        
        # Stratified Split
        train_df, val_df, test_df = stratified_split(df_finsen, 'Tag')
        
        # Save Splits
        train_df.to_csv(os.path.join(finsen_dir, "train_finsen.csv"), index=False)
        val_df.to_csv(os.path.join(finsen_dir, "val_finsen.csv"), index=False)
        test_df.to_csv(os.path.join(finsen_dir, "test_finsen.csv"), index=False)
        print(f"Finsen Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # NVDM (BoW)
        X_train, X_val, X_test, vec = create_bow_matrices(
            train_df['processed_content'], 
            val_df['processed_content'], 
            test_df['processed_content']
        )
        save_numpy_arrays(finsen_dir, "finsen_bow", {
            "train": X_train.toarray(), # Converting sparse to dense for NPY save (check memory if huge, but usually OK for datasets of this size)
            "val": X_val.toarray(),
            "test": X_test.toarray()
        })
        # Save vocab
        import json
        with open(os.path.join(finsen_dir, "finsen_vocab.json"), "w") as f:
            json.dump(vec.vocabulary_, f)
            
        # FinBERT
        print("Tokenizing Finsen for FinBERT...")
        tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        
        train_ids, train_mask = create_finbert_inputs(train_df['processed_content'], tokenizer)
        val_ids, val_mask = create_finbert_inputs(val_df['processed_content'], tokenizer)
        test_ids, test_mask = create_finbert_inputs(test_df['processed_content'], tokenizer)
        
        save_numpy_arrays(finsen_dir, "finsen_finbert", {
            "train_ids": train_ids, "train_mask": train_mask,
            "val_ids": val_ids, "val_mask": val_mask,
            "test_ids": test_ids, "test_mask": test_mask
        })
        
    else:
        print(f"Finsen preprocessed file not found: {finsen_path}")

    # ---------------------------------------------------------
    # 2. Processing StockEmotions Dataset
    # ---------------------------------------------------------
    print("\nProcessing StockEmotions Dataset...")
    # Load already split files
    train_path = os.path.join(stockemo_dir, "train_stockemo_preprocessed.csv")
    val_path = os.path.join(stockemo_dir, "val_stockemo_preprocessed.csv")
    test_path = os.path.join(stockemo_dir, "test_stockemo_preprocessed.csv")
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path).dropna(subset=['processed_content'])
        val_df = pd.read_csv(val_path).dropna(subset=['processed_content'])
        test_df = pd.read_csv(test_path).dropna(subset=['processed_content'])
        
        print(f"StockEmotions Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # NVDM (BoW)
        X_train, X_val, X_test, vec = create_bow_matrices(
            train_df['processed_content'], 
            val_df['processed_content'], 
            test_df['processed_content']
        )
        save_numpy_arrays(stockemo_dir, "stockemo_bow", {
            "train": X_train.toarray(),
            "val": X_val.toarray(),
            "test": X_test.toarray()
        })
        # Save vocab
        with open(os.path.join(stockemo_dir, "stockemo_vocab.json"), "w") as f:
            json.dump(vec.vocabulary_, f)

        # FinBERT
        print("Tokenizing StockEmotions for FinBERT...")
        # Re-use tokenizer
        tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        
        train_ids, train_mask = create_finbert_inputs(train_df['processed_content'], tokenizer)
        val_ids, val_mask = create_finbert_inputs(val_df['processed_content'], tokenizer)
        test_ids, test_mask = create_finbert_inputs(test_df['processed_content'], tokenizer)
        
        save_numpy_arrays(stockemo_dir, "stockemo_finbert", {
            "train_ids": train_ids, "train_mask": train_mask,
            "val_ids": val_ids, "val_mask": val_mask,
            "test_ids": test_ids, "test_mask": test_mask
        })
        
    else:
        print("StockEmotions preprocessed files not found.")

    print("\nModule 1.2 Data Preparation Completed.")

if __name__ == "__main__":
    main()
