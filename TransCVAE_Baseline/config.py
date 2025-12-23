CONFIG = {
    "audio_dim": 128,         # mel-spectrogram 次元
    "motion_dim": 78,         # SMPL motion 次元（poses(72) + Th(3) + Rh(3) = 78）
    "lyrics_dim": 768,        # 投影後の歌詞埋め込み次元
    
    # Transformer パラメータ
    "d_model": 256,           # Transformer の内部次元
    "nhead": 4,               # Multi-head attention のヘッド数
    "num_layers": 2,          # Transformer レイヤー数
    "dim_feedforward": 512,   # FFN の隠れ層次元
    "latent_dim": 128,        # VAE 潜在次元
    "seq_len": 180,           # シーケンス長（6秒 @30fps）
    
    # 学習パラメータ
    "lr": 1e-3,
    "kld_weight": 1e-3,
    "batch_size": 16,
}
