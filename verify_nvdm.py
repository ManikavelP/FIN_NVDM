import torch
from nvdm import NVDM
import numpy as np

def verify_architecture():
    print("Verifying NVDM Architecture...")
    vocab_size = 11730 # From Module 1.3 check
    model = NVDM(vocab_size=vocab_size, hidden_dim=500, latent_dim=40)
    
    print(model)
    
    batch_size = 32
    dummy_input = torch.randn(batch_size, vocab_size)
    
    recon_logits, mu, logvar = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Recon Logits Shape: {recon_logits.shape}")
    print(f"Mu Shape: {mu.shape}")
    print(f"LogVar Shape: {logvar.shape}")
    
    if recon_logits.shape == (batch_size, vocab_size):
        print("SUCCESS: Reconstruction shape matches input.")
    else:
        print("FAILURE: Reconstruction shape mismatch.")
        
    if mu.shape == (batch_size, 40) and logvar.shape == (batch_size, 40):
        print("SUCCESS: Latent shape matches configuration (K=40).")
    else:
        print("FAILURE: Latent shape mismatch.")

if __name__ == "__main__":
    verify_architecture()
