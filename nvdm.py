import torch
import torch.nn as nn
import torch.nn.functional as F

class NVDM(nn.Module):
    """
    Neural Variational Document Model (NVDM)
    As specified in Module 2.
    """
    def __init__(self, vocab_size, hidden_dim=500, latent_dim=40):
        super(NVDM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # -------------------------------------------------------
        # 2.2 Encoder Network (Inference Network) q(z|d)
        # -------------------------------------------------------
        # Input: BoW Vector (V) -> Hidden (500)
        self.encoder_fc = nn.Linear(vocab_size, hidden_dim)
        
        # Latent Parameters: Hidden (500) -> Latent (40)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # log(sigma^2)

        # -------------------------------------------------------
        # 2.3 Decoder Network (Generative Network) p(d|z)
        # -------------------------------------------------------
        # Latent (40) -> Logits (V)
        # Weights initialized randomly (default PyTorch init is fine/random)
        self.decoder_fc = nn.Linear(latent_dim, vocab_size)

    def encode(self, x):
        """
        Maps input x (BoW) to latent distribution parameters.
        """
        # Activation: ReLU
        h = F.relu(self.encoder_fc(x))
        
        # Latent params
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Sampling: z = mu + sigma * epsilon
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu # Deterministic in eval mode

    def decode(self, z):
        """
        Maps latent vector z to word logits.
        """
        # Logits = Linear(z)
        # Softmax is applied during loss calculation (often more stable)
        # But for probability output we can apply softmax here if needed.
        return self.decoder_fc(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar
