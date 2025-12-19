import torch
import torch.nn as nn
from transformers import BertModel

class FinBERTClassifier(nn.Module):
    def __init__(self, num_classes=12, frozen=False):
        super(FinBERTClassifier, self).__init__()
        # Load Pre-trained FinBERT
        self.bert = BertModel.from_pretrained('ProsusAI/finbert')
        
        # Freezing option (usually we fine-tune all, but good to have)
        if frozen:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        # Classification Head
        # ProsusAI/finbert hidden size is 768
        self.classifier = nn.Linear(768, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # Pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract [CLS] token representation (pooler_output or last_hidden_state[:, 0])
        # pooler_output usually has a dense+tanh on top of CLS, usually better for classification.
        pooled_output = outputs.pooler_output
        
        # Dropout
        x = self.dropout(pooled_output)
        
        # Logits
        logits = self.classifier(x)
        
        return logits
