import torch
import numpy as np
import os
import json
from nvdm import NVDM
import gensim
from gensim.models.coherencemodel import CoherenceModel

def evaluate():
    print("--- Evaluating NVDM ---")
    base_dir = r"d:\Documents1\Project"
    finsen_dir = os.path.join(base_dir, "kaggle_finsen_dataset")
    
    # Paths
    model_path = os.path.join(finsen_dir, "nvdm_finsen.pth")
    interrupted_path = os.path.join(finsen_dir, "nvdm_finsen_interrupted.pth")
    vocab_path = os.path.join(finsen_dir, "finsen_vocab.json")
    
    # Load Model
    vocab_size = 11730 # Hardcoded or load from vocab
    if os.path.exists(model_path):
        load_path = model_path
    elif os.path.exists(interrupted_path):
        print("Loading interrupted model checkpoint...")
        load_path = interrupted_path
    else:
        print("No model found.")
        return

    # Load Vocab
    with open(vocab_path, 'r') as f:
        vocab_dict = json.load(f)
    # Invert vocab: id -> word
    id2word = {v: k for k, v in vocab_dict.items()}
    
    model = NVDM(vocab_size=len(vocab_dict), hidden_dim=500, latent_dim=40)
    model.load_state_dict(torch.load(load_path, map_location='cpu'))
    model.eval()
    
    # Task 1: Topic Extraction
    # Decoder weights: (Vocab, Latent) or (Latent, Vocab)?
    # nn.Linear(latent, vocab) -> weight shape is (Vocab, Latent) in PyTorch? 
    # Let's check: Linear(in, out). Waight is (out, in).
    # So decoder.weight is (Vocab, Latent).
    print("Extracting Topics...")
    decoder_weight = model.decoder_fc.weight.data.cpu().numpy() # (V, K)
    
    topics = []
    top_n = 10
    
    for k in range(40):
        # We want top words for topic k.
        # The weight W_w maps z to words. 
        # Logits = W z + b. 
        # Strong positive weight from topic k to word w means z_k contributes to w.
        # So we look at column k of W? 
        # weight is (Out, In) = (V, K). 
        # So for topic k (input dimension k), we look at column k? 
        # W has shape (V, 40). We want column k.
        topic_impt = decoder_weight[:, k]
        top_indices = topic_impt.argsort()[::-1][:top_n]
        top_words = [id2word[i] for i in top_indices]
        topics.append(top_words)
        # print(f"Topic {k}: {top_words}")
        
    # Save Topics
    with open(os.path.join(finsen_dir, "topics_k40.txt"), "w") as f:
        for i, t in enumerate(topics):
            f.write(f"Topic {i}: {' '.join(t)}\n")
    print("Topics saved to topics_k40.txt")

    # Task 2: Coherence (c_v)
    # We need texts (list of list of tokens) to compute coherence.
    # Gensim requires tokenized texts.
    # We can load a subset of original texts or re-construct reference corpus?
    # Coherence c_v calculates probabilities from a reference corpus (sliding window).
    # We should use the validation set 'original' texts (if available) or training texts.
    # Loading raw text is slow. Let's try to load 'val_finsen.csv' content column.
    
    print("Calculating Coherence...")
    import pandas as pd
    from gensim.corpora import Dictionary
    try:
        val_df = pd.read_csv(os.path.join(finsen_dir, "val_finsen.csv"))
        # Tokenize simply for reference
        texts = [doc.split() for doc in val_df['processed_content'].astype(str).tolist()]
        
        # Create Dictionary from texts (Reference corpus)
        # Ideally we should use the same vocab as model, but Coherence uses this dictionary to compute probabilities in the reference corpus.
        # So it should be built from the reference texts suitable for c_v.
        dictionary = Dictionary(texts)
        
        # Calculate Coherence
        cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = cm.get_coherence()
        print(f"Coherence Score (c_v): {coherence:.4f}")
        
    except Exception as e:
        print(f"Coherence calculation failed: {e}")

    # Task 3: Perplexity
    print("Calculating Perplexity on Validation Set...")
    try:
        val_path = os.path.join(finsen_dir, "finsen_bow_test_resampled.npz") 
        # Actually we should use val_finsen.csv vectorization which we saved as finsen_bow_val.npy (dense usually, or sparse if we updated prepare_data, 
        # wait, prepare_data saved 'finsen_bow_val.npy' as dense array (using .toarray()). 
        # Let's check format. `save_numpy_arrays` saved .npy.
        # But `balance_data` saved .npz for resampled train.
        # Validation set was NOT resampled (correctly). So it is in `finsen_bow_val.npy`.
        val_bow_path = os.path.join(finsen_dir, "finsen_bow_val.npy")
        
        if os.path.exists(val_bow_path):
             val_data = np.load(val_bow_path)
             # Dense array
             val_tensor = torch.FloatTensor(val_data)
        else:
             print("Validation BoW not found.")
             return

        model.eval()
        with torch.no_grad():
            # Process in batches to avoid OOM
            batch_size = 100
            total_nll = 0
            total_words = 0
            
            for i in range(0, len(val_tensor), batch_size):
                batch = val_tensor[i:i+batch_size]
                if batch.max() == 0: continue # Skip empty?
                
                recon_logits, mu, logvar = model(batch)
                log_probs = torch.nn.functional.log_softmax(recon_logits, dim=1)
                
                # NLL = - sum(x * log_p)
                # But for perplexity we usually sum over all docs then div by total words.
                # Perplexity = exp( Total_NLL / Total_Words )
                
                nll = -(batch * log_probs).sum().item()
                n_words = batch.sum().item()
                
                total_nll += nll
                total_words += n_words
            
            if total_words > 0:
                ppl = np.exp(total_nll / total_words)
                print(f"Perplexity: {ppl:.4f}")
            else:
                print("Total words is 0.")
                
    except Exception as e:
        print(f"Perplexity calculation failed: {e}")

if __name__ == "__main__":
    evaluate()
