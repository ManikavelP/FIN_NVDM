import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse
import os
from nvdm import NVDM
import torch.nn.functional as F
import json

def print_top_words(model, id2word, n_topics=5, n_words=10):
    model.eval()
    print(f"\n--- Top Words for first {n_topics} Topics ---")
    # Decoder: Linear(Latent, Vocab). Weight: (Vocab, Latent)
    decoder_weight = model.decoder_fc.weight.data.cpu().numpy()
    
    for k in range(n_topics):
        topic_col = decoder_weight[:, k]
        top_indices = topic_col.argsort()[::-1][:n_words]
        top_words = [id2word[i] for i in top_indices]
        print(f"Topic {k}: {', '.join(top_words)}")
    print("---------------------------------------------\n")
    model.train()

class SparseBowDataset(Dataset):
    def __init__(self, npz_path):
        print(f"Loading {npz_path}...")
        self.data = scipy.sparse.load_npz(npz_path)
        # Ensure it's CSR for fast row slicing
        self.data = self.data.tocsr()
        self.n_samples = self.data.shape[0]
        self.vocab_size = self.data.shape[1]
        print(f"Loaded {self.n_samples} samples, {self.vocab_size} vocab features.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Determine slice safely
        row = self.data[idx]
        # Convert to dense numpy array then tensor
        # Row is (1, V), squeeze to (V,)
        return torch.FloatTensor(row.toarray()).squeeze(0)

def loss_function(recon_logits, x, mu, logvar):
    """
    ELBO = NLL + KL
    NLL = - term_freq * log_prob
    """
    # x is BoW counts.
    # recon_logits is unnormalized logits.
    log_probs = F.log_softmax(recon_logits, dim=1)
    
    # NLL: Sum over vocabulary
    # x * log_probs gives the log-likelihood of each word weighted by its count
    # Sum over vocab dim (dim 1)
    # Mean over batch dim (standard practice, or sum if strictly variational)
    # We maintain "mean over batch" to be optimizer-friendly
    recon_loss = -(x * log_probs).sum(dim=1).mean()

    # KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Sum over latent dim (dim 1)
    # Mean over batch
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    return recon_loss + kl_loss, recon_loss, kl_loss

def train(model, train_loader, optimizer, epochs=10, device='cpu', save_dir=None, id2word=None):
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_logits, mu, logvar = model(data)
            loss, recon, kl = loss_function(recon_logits, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} (Recon: {recon.item():.4f}, KL: {kl.item():.4f})")
        
        avg_loss = total_loss / len(train_loader)
        print(f"=== Epoch {epoch+1} Average Loss: {avg_loss:.4f} ===")
        
        # Monitor Topics
        if id2word:
            print_top_words(model, id2word, n_topics=5, n_words=8)
        
        # Save Best Model Logic
        if save_dir and avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "nvdm_finsen_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved to {best_path} (Loss: {best_loss:.4f})")

def main():
    # Detect current directory (compatible with script and notebook/colab)
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
        
    finsen_dir = os.path.join(base_dir, "kaggle_finsen_dataset")
    
    # Paths
    train_path = os.path.join(finsen_dir, "finsen_bow_train_resampled.npz")
    
    if not os.path.exists(train_path):
        print(f"Training file not found: {train_path}")
        return

    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 15 # Increased for full training
    LR = 1e-3
    LATENT_DIM = 40
    HIDDEN_DIM = 500
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Data
    dataset = SparseBowDataset(train_path)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = NVDM(vocab_size=dataset.vocab_size, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    model.to(DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Load Vocab for monitoring
    vocab_path = os.path.join(finsen_dir, "finsen_vocab.json")
    id2word = None
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
            id2word = {v: k for k, v in vocab.items()}
    else:
        print("Warning: Vocab file not found. Topic monitoring disabled.")

    # Training Loop
    try:
        train(model, train_loader, optimizer, epochs=EPOCHS, device=DEVICE, save_dir=finsen_dir, id2word=id2word)
        
        # Save Model
        save_path = os.path.join(finsen_dir, "nvdm_finsen.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
        save_path = os.path.join(finsen_dir, "nvdm_finsen_interrupted.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
