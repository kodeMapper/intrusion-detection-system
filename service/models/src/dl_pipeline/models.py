import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out shape: (Batch, Seq_Len, Hidden_Dim)
        attn_weights = self.attention(lstm_out) # (Batch, Seq_Len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        # Weighted sum along Seq_Len
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # (Batch, Hidden_Dim)
        return context_vector

class FlowLSTM(nn.Module):
    """
    Bidirectional LSTM with Attention and LayerNorm for learning temporal 
    relationships across consecutive network flows.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 6):
        super(FlowLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Bidirectional doubles the hidden_dim
        lstm_out_dim = hidden_dim * 2
        
        self.attention = Attention(lstm_out_dim)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, Input_Dim)
        lstm_out, _ = self.lstm(x)
        
        # Apply Attention mechanism over the sequence
        context = self.attention(lstm_out)
        
        # Apply LayerNorm
        context = self.layer_norm(context)
        
        # Output raw logits
        logits = self.fc(context)
        return logits


class AnomalyAutoencoder(nn.Module):
    """
    Unsupervised Autoencoder trained to reconstruct normal network traffic.
    Expanded capacity with BatchNorm and LeakyReLU to prevent underfitting.
    """
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super(AnomalyAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, latent_dim),
            nn.LeakyReLU(0.1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
