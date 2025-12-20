import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        return self.transformer(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)


class CVAEEncoder(nn.Module):
    def __init__(self, audio_dim: int, motion_dim: int, lyrics_dim: int, d_model: int, 
                 nhead: int, num_layers: int, dim_feedforward: int, latent_dim: int):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.lyrics_proj = nn.Linear(lyrics_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward)
        
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, audio, motion, lyrics):
        # (B*T, audio_dim), (B*T, motion_dim), (B*T, lyrics_dim) → (B, T, d_model)
        B_T = audio.size(0)
        T = 180  # シーケンス長
        B = B_T // T
        
        audio_emb = self.audio_proj(audio).view(B, T, -1)
        motion_emb = self.motion_proj(motion).view(B, T, -1)
        lyrics_emb = self.lyrics_proj(lyrics).view(B, T, -1)
        
        # 条件と target を結合
        x = audio_emb + motion_emb + lyrics_emb  # 要素単位で統合
        x = self.pos_encoding(x)
        
        # Transformer Encoder
        encoded = self.transformer(x)  # (B, T, d_model)
        
        # 平均プーリングで全体の特徴を取得
        pooled = encoded.mean(dim=1)  # (B, d_model)
        mu = self.fc_mu(pooled)  # (B, latent_dim)
        logvar = self.fc_logvar(pooled)  # (B, latent_dim)
        
        return mu, logvar


class CVAEDecoder(nn.Module):
    def __init__(self, audio_dim: int, motion_dim: int, lyrics_dim: int, d_model: int,
                 nhead: int, num_layers: int, dim_feedforward: int, latent_dim: int, seq_len: int = 180):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.lyrics_proj = nn.Linear(lyrics_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer Decoder（条件を memory として使用）
        self.transformer = TransformerDecoder(d_model, nhead, num_layers, dim_feedforward)
        
        self.audio_out = nn.Linear(d_model, audio_dim)

    def forward(self, z, motion, lyrics):
        # (B, latent_dim), (B*T, motion_dim), (B*T, lyrics_dim)
        B_T = motion.size(0)
        T = self.seq_len
        B = B_T // T
        
        motion_emb = self.motion_proj(motion).view(B, T, -1)
        lyrics_emb = self.lyrics_proj(lyrics).view(B, T, -1)
        
        # memory: 条件を Transformer decoder memory として使用
        memory = motion_emb + lyrics_emb  # (B, T, d_model)
        memory = self.pos_encoding(memory)
        
        # decoder target: z をシーケンス長にまで拡張
        z_expanded = self.z_proj(z).unsqueeze(1).expand(B, T, -1)  # (B, T, d_model)
        tgt = self.pos_encoding(z_expanded)
        
        # Transformer Decoder
        decoded = self.transformer(tgt, memory)  # (B, T, d_model)
        
        # Audio 出力
        audio_out = self.audio_out(decoded)  # (B, T, audio_dim)
        audio_out = audio_out.view(B * T, -1)  # (B*T, audio_dim)
        
        return audio_out


class TransCVAE(nn.Module):
    """
    Transformer ベースの Conditional VAE。
    Motion + Lyrics を条件に Audio を生成する。
    """
    def __init__(self, audio_dim: int, motion_dim: int, lyrics_dim: int,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 512, latent_dim: int = 128, seq_len: int = 180):
        super().__init__()
        self.encoder = CVAEEncoder(audio_dim, motion_dim, lyrics_dim, d_model, nhead, num_layers, dim_feedforward, latent_dim)
        self.decoder = CVAEDecoder(audio_dim, motion_dim, lyrics_dim, d_model, nhead, num_layers, dim_feedforward, latent_dim, seq_len)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, audio, motion, lyrics):
        mu, logvar = self.encoder(audio, motion, lyrics)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, motion, lyrics)
        return recon, mu, logvar


def kld_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


def recon_loss_l1(recon, target):
    return F.l1_loss(recon, target)
