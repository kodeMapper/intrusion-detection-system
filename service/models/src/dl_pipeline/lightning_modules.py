import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class AutoencoderDataModule(L.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size=256):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train.copy()),
            torch.LongTensor(self.y_train.copy())
        )
        self.val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val.copy()),
            torch.LongTensor(self.y_val.copy())
        )

    def train_dataloader(self):
        # We only need the features for AE training, but we keep labels for consistency
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

class AnomalyAutoencoderLightning(L.LightningModule):
    def __init__(self, input_dim=198, latent_dim=8, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(128, input_dim)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x, _ = batch
        encoded = self.encoder(x)
        x_hat = self.decoder(encoded)
        
        mse_loss = self.criterion(x_hat, x)
        l1_loss = torch.mean(torch.abs(encoded))
        loss = mse_loss + 1e-4 * l1_loss
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

import torch.nn.functional as F

class TemporalTransformerClassifier(L.LightningModule):
    """
    Supervised temporal classifier for flow sequences shaped (batch, seq_len, input_size).
    Used for both Stage 1 binary attack gating and Stage 2 attack classification.
    """
    def __init__(
        self,
        input_size: int = 198,
        seq_len: int = 10,
        num_classes: int = 2,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        classifier_hidden: int = 128,
        dropout: float = 0.15,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        focal_gamma: float = 2.0,
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.input_proj = nn.Linear(input_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

        if class_weights is None:
            class_weights = torch.ones(num_classes, dtype=torch.float32)
        else:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        tokens = self.input_proj(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.size(1), :]
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded[:, 0])
        return self.classifier(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(x), dim=1)

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        class_weights = self.class_weights
        if class_weights.numel() != self.hparams.num_classes:
            class_weights = None

        ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
        gamma = float(self.hparams.focal_gamma)
        if gamma > 0.0:
            pt = torch.exp(-ce)
            ce = ((1.0 - pt) ** gamma) * ce
        return ce.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

class TemporalOneClassVAE(L.LightningModule):
    """
    Normal-only temporal VAE with a one-class latent objective.

    forward() returns only the reconstruction tensor so legacy autoencoder
    evaluation and ONNX export paths can keep treating the module as X -> X.
    """
    def __init__(
        self,
        input_size: int = 198,
        seq_len: int = 10,
        d_model: int = 128,
        latent_dim: int = 16,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        decoder_hidden: int = 64,
        decoder_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        beta_max: float = 0.02,
        kl_warmup_epochs: int = 20,
        center_loss_weight: float = 0.2,
        score_weights: tuple = (0.45, 0.35, 0.20),
        calibration_eps: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_proj = nn.Linear(input_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, decoder_hidden)
        self.time_embedding = nn.Parameter(torch.zeros(1, seq_len, decoder_hidden))
        decoder_layers = max(1, int(decoder_layers))
        decoder_modules = [
            nn.LayerNorm(decoder_hidden),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(decoder_layers - 1):
            decoder_modules.extend(
                [
                    nn.LayerNorm(decoder_hidden),
                    nn.Linear(decoder_hidden, decoder_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        decoder_modules.append(nn.Linear(decoder_hidden, input_size))
        self.decoder = nn.Sequential(*decoder_modules)

        self.register_buffer("latent_center", torch.zeros(latent_dim))
        self.register_buffer("score_median", torch.zeros(3))
        self.register_buffer("score_iqr", torch.ones(3))
        self.register_buffer("score_weights", torch.tensor(score_weights, dtype=torch.float32))
        self.register_buffer("is_score_calibrated", torch.tensor(False, dtype=torch.bool))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        tokens = self.input_proj(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.size(1), :]
        encoded = self.encoder(tokens)
        cls_state = self.encoder_norm(encoded[:, 0])
        mu = self.mu_head(cls_state)
        logvar = torch.clamp(self.logvar_head(cls_state), min=-8.0, max=6.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.latent_to_hidden(z).unsqueeze(1)
        hidden = hidden.expand(-1, self.hparams.seq_len, -1)
        hidden = hidden + self.time_embedding[:, : self.hparams.seq_len, :]
        return self.decoder(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return self.decode(mu)

    def _loss_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        center_loss = torch.mean(torch.sum((mu - self.latent_center.unsqueeze(0)).pow(2), dim=1))
        return recon_loss, kl_loss, center_loss, recon

    def _kl_weight(self) -> float:
        warmup = max(1, int(self.hparams.kl_warmup_epochs))
        progress = min(1.0, float(self.current_epoch + 1) / float(warmup))
        return float(self.hparams.beta_max) * progress

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_loss, kl_loss, center_loss, _ = self._loss_components(x)
        kl_weight = self._kl_weight()
        loss = recon_loss + kl_weight * kl_loss + self.hparams.center_loss_weight * center_loss
        self.log("train_recon_loss", recon_loss, on_epoch=True, prog_bar=True)
        self.log("train_kl_loss", kl_loss, on_epoch=True)
        self.log("train_center_loss", center_loss, on_epoch=True)
        self.log("train_kl_weight", kl_weight, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon_loss, kl_loss, center_loss, _ = self._loss_components(x)
        loss = recon_loss + self.hparams.beta_max * kl_loss + self.hparams.center_loss_weight * center_loss
        self.log("val_recon_loss", recon_loss, on_epoch=True, prog_bar=True)
        self.log("val_kl_loss", kl_loss, on_epoch=True)
        self.log("val_center_loss", center_loss, on_epoch=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def anomaly_components(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        recon = self.decode(mu)
        recon_mse = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))
        center_dist = (mu - self.latent_center.unsqueeze(0)).pow(2).mean(dim=1)
        kl_surprise = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_surprise = kl_surprise / max(1, self.hparams.latent_dim)
        return torch.stack([recon_mse, center_dist, kl_surprise], dim=1)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        components = self.anomaly_components(x)
        if bool(self.is_score_calibrated.item()):
            scaled = (components - self.score_median.unsqueeze(0)) / (
                self.score_iqr.unsqueeze(0) + self.hparams.calibration_eps
            )
            scaled = torch.clamp(scaled, min=0.0)
        else:
            scaled = components
        weights = self.score_weights / torch.clamp(
            self.score_weights.sum(), min=self.hparams.calibration_eps
        )
        return torch.sum(scaled * weights.unsqueeze(0), dim=1)

    def set_score_weights(self, score_weights):
        weights = torch.tensor(
            score_weights,
            dtype=self.score_weights.dtype,
            device=self.score_weights.device,
        )
        if weights.numel() != 3:
            raise ValueError("score_weights must contain exactly 3 values.")
        if torch.any(weights < 0) or torch.sum(weights) <= 0:
            raise ValueError("score_weights must be non-negative and sum to a positive value.")
        self.score_weights.copy_(weights)

    @torch.no_grad()
    def initialize_center(self, dataloader, device: torch.device):
        self.eval()
        mus = []
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            mu, _ = self.encode(x)
            mus.append(mu.detach().cpu())
        if not mus:
            raise ValueError("Cannot initialize latent center from an empty dataloader.")
        center = torch.cat(mus, dim=0).mean(dim=0)
        center[(center.abs() < 1e-4) & (center < 0)] = -1e-4
        center[(center.abs() < 1e-4) & (center >= 0)] = 1e-4
        self.latent_center.copy_(center.to(self.latent_center.device))

    @torch.no_grad()
    def calibrate_score(self, dataloader, device: torch.device) -> dict:
        self.eval()
        all_components = []
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            all_components.append(self.anomaly_components(x).detach().cpu())
        if not all_components:
            raise ValueError("Cannot calibrate anomaly score from an empty dataloader.")

        components = torch.cat(all_components, dim=0)
        q25 = torch.quantile(components, 0.25, dim=0)
        median = torch.quantile(components, 0.50, dim=0)
        q75 = torch.quantile(components, 0.75, dim=0)
        iqr = torch.clamp(q75 - q25, min=self.hparams.calibration_eps)

        self.score_median.copy_(median.to(self.score_median.device))
        self.score_iqr.copy_(iqr.to(self.score_iqr.device))
        self.is_score_calibrated.copy_(torch.tensor(True, device=self.is_score_calibrated.device))
        return {
            "median": median.numpy().tolist(),
            "iqr": iqr.numpy().tolist(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

class DenoisingAugment:
    """Apply during training only — never at inference."""
    def __init__(self, noise_factor=0.05, mask_prob=0.15):
        self.noise_factor = noise_factor
        self.mask_prob = mask_prob

    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise

    def random_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out mask_prob fraction of features per sample."""
        mask = torch.bernoulli(
            torch.full_like(x, 1 - self.mask_prob)
        )
        return x * mask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gaussian_noise(x)
        x = self.random_mask(x)
        return x

class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, latent_dim: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            nn.Tanh(),  # Bounded activation → limits representational capacity
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, _) = self.lstm(x)
        # Use last hidden state of top layer
        h_last = h_n[-1]                  # (batch, hidden_size)
        h_norm = self.layer_norm(h_last)
        z = self.bottleneck(h_norm)        # (batch, latent_dim)
        return z

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int,
                 num_layers: int, output_size: int,
                 seq_len: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.expand = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        # z: (batch, latent_dim)
        h0 = self.expand(z).unsqueeze(0).repeat(
            self.lstm.num_layers, 1, 1
        )  # (num_layers, batch, hidden_size)
        c0 = torch.zeros_like(h0)
        # Repeat latent vector across time steps
        z_seq = self.expand(z).unsqueeze(1).repeat(
            1, self.seq_len, 1
        )  # (batch, seq_len, hidden_size)
        out, _ = self.lstm(z_seq, (h0, c0))
        out = self.layer_norm(out)
        return self.output_proj(out)      # (batch, seq_len, output_size)

class LSTMAutoencoder(L.LightningModule):
    def __init__(
        self,
        input_size: int = 198,
        seq_len: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        latent_dim: int = 32,
        dropout: float = 0.2,
        lambda_sparse: float = 1e-5,
        noise_factor: float = 0.05,
        mask_prob: float = 0.15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = LSTMEncoder(
            input_size, hidden_size, num_layers, latent_dim, dropout
        )
        self.decoder = LSTMDecoder(
            latent_dim, hidden_size, num_layers, input_size, seq_len, dropout
        )
        self.augment = DenoisingAugment(noise_factor, mask_prob)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def _shared_step(self, x_clean, stage: str):
        if stage == "train":
            x_in = self.augment(x_clean)
        else:
            x_in = x_clean

        recon, z = self(x_in)
        recon_loss = F.mse_loss(recon, x_clean)  # Always reconstruct CLEAN
        sparse_reg = self.hparams.lambda_sparse * z.abs().mean()
        total_loss = recon_loss + sparse_reg

        self.log(f"{stage}_recon_loss", recon_loss, prog_bar=True)
        self.log(f"{stage}_loss", total_loss, prog_bar=(stage == "train"))
        return total_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        return self._shared_step(x, "train")

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        return self._shared_step(x, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

class MemoryModule(nn.Module):
    """
    Discrete Memory matrix to constrain continuous latent representations.
    Forces the network to reconstruct using only a sparse combination of normal prototypes.
    """
    def __init__(self, mem_dim: int, fea_dim: int, shrink_thres: float = 0.0025, alpha: float = 15.0):
        super(MemoryModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, z):
        # z: (batch, fea_dim)
        # L2-normalize query z and memory items to compute cosine similarity
        z_norm = F.normalize(z, p=2, dim=1)  # (batch, fea_dim)
        mem_norm = F.normalize(self.weight, p=2, dim=1)  # (mem_dim, fea_dim)
        
        # Compute attention weights using cosine similarity and scale
        att_weight = F.linear(z_norm, mem_norm) * self.alpha  # (batch, mem_dim)
        att_weight = F.softmax(att_weight, dim=1) # (batch, mem_dim)
        
        # Hard shrinkage to force sparse memory addressing
        if self.shrink_thres > 0:
            att_weight = F.relu(att_weight - self.shrink_thres)
            # Re-normalize
            att_weight = att_weight / (torch.sum(att_weight, dim=1, keepdim=True) + 1e-12)
            
        z_hat = F.linear(att_weight, self.weight.t()) # (batch, fea_dim)
        return z_hat, att_weight

class MemoryLSTMAutoencoderLightning(L.LightningModule):
    def __init__(
        self,
        input_size: int = 198,
        seq_len: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        latent_dim: int = 32,
        mem_dim: int = 50,
        shrink_thres: float = 0.0025,
        alpha: float = 15.0,
        dropout: float = 0.2,
        lambda_entropy: float = 0.0002,
        noise_factor: float = 0.05,
        mask_prob: float = 0.15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = LSTMEncoder(
            input_size, hidden_size, num_layers, latent_dim, dropout
        )
        self.memory = MemoryModule(mem_dim, latent_dim, shrink_thres, alpha)
        self.decoder = LSTMDecoder(
            latent_dim, hidden_size, num_layers, input_size, seq_len, dropout
        )
        self.augment = DenoisingAugment(noise_factor, mask_prob)

    def forward(self, x):
        z = self.encoder(x)
        z_hat, att_weight = self.memory(z)
        return self.decoder(z_hat), z, att_weight

    def _shared_step(self, x_clean, stage: str):
        if stage == "train":
            x_in = self.augment(x_clean)
        else:
            x_in = x_clean

        recon, z, att_weight = self(x_in)
        recon_loss = F.mse_loss(recon, x_clean)  # Reconstruct clean sequence
        
        # Entropy loss to encourage sparse addressing (we want it to use very few prototypes)
        entropy_loss = -torch.sum(att_weight * torch.log(att_weight + 1e-12), dim=1).mean()
        
        total_loss = recon_loss + self.hparams.lambda_entropy * entropy_loss

        self.log(f"{stage}_recon_loss", recon_loss, prog_bar=True)
        self.log(f"{stage}_entropy_loss", entropy_loss, prog_bar=(stage == "train"))
        self.log(f"{stage}_loss", total_loss, prog_bar=(stage == "train"))
        return total_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        return self._shared_step(x, "train")

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        return self._shared_step(x, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
