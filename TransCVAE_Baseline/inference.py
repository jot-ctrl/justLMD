# TransCVAE_Baseline/inference.py
import os
import inspect
import torch
import soundfile as sf

import torchaudio
import numpy as np
import librosa
import torch.nn.functional as F

from src.models.architectures.transcvae import TransCVAE
from data_loader import get_2022_dataloader
from config import CONFIG

_VOCODER = None


def load_vocoder():
    global _VOCODER
    if _VOCODER is None:
        try:
            from vocos import Vocos
        except ImportError:
            raise ImportError("vocos が未インストールです: pip install vocos")
        _VOCODER = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
    return _VOCODER


def _as_bt_mel(x: torch.Tensor, seq_len: int, n_mels: int = 128) -> torch.Tensor:
    """
    x:
      - (B, T, n_mels) -> そのまま
      - (T, n_mels)    -> (1, T, n_mels)
      - (B*T, n_mels)  -> (B, T, n_mels) に復元
    """
    if x.dim() == 3:
        return x
    if x.dim() == 2:
        if x.size(-1) != n_mels:
            raise RuntimeError(f"Expected last dim n_mels={n_mels}, got {tuple(x.shape)}")
        # (T, n_mels)
        if x.size(0) == seq_len:
            return x.unsqueeze(0)
        # (B*T, n_mels)
        if x.size(0) % seq_len == 0:
            B = x.size(0) // seq_len
            return x.reshape(B, seq_len, n_mels)
    raise RuntimeError(f"Unsupported mel shape: {tuple(x.shape)}")


def mel_to_audio(mel_db: torch.Tensor, target_sr: int = 18000):
    """
    mel_db: (T, n_mels_model) を想定（今は n_mels_model=128）
    vocoder側の n_mels に合わせて周波数軸をリサンプルしてから decode する
    """
    if mel_db.dim() != 2:
        raise RuntimeError(f"mel_to_audio expects (T, n_mels), got {tuple(mel_db.shape)}")

    vocoder = load_vocoder()

    # vocoder が期待する mel bin 数を取得（Vocos は backbone.embed が Conv1d）
    try:
        expected_mels = int(vocoder.backbone.embed.in_channels)
    except Exception:
        expected_mels = 100  # fallback

    # (T, M) -> (1, M, T)
    mel_np = mel_db.detach().cpu().numpy()
    mel_power = librosa.db_to_power(mel_np)  # ※モデル出力が dB 前提。後述参照
    mel_tensor = torch.from_numpy(mel_power).float().transpose(0, 1).unsqueeze(0)  # (1, M, T)

    # 128 -> 100 など、mel bin が違う場合は周波数軸を補間
    if mel_tensor.size(1) != expected_mels:
        # (B, M, T) -> (B, 1, M, T) として 2D 補間
        mel_tensor_2d = mel_tensor.unsqueeze(1)  # (1, 1, M, T)
        mel_tensor_2d = F.interpolate(
            mel_tensor_2d,
            size=(expected_mels, mel_tensor.size(-1)),
            mode="bilinear",
            align_corners=False,
        )
        mel_tensor = mel_tensor_2d.squeeze(1)  # (1, expected_mels, T)

    mel_tensor = mel_tensor.to(next(vocoder.parameters()).device)

    with torch.no_grad():
        audio_24k = vocoder.decode(mel_tensor)  # (1, samples) @24kHz

    if target_sr != 24000:
        audio_18k = torchaudio.functional.resample(audio_24k, orig_freq=24000, new_freq=target_sr)
        return audio_18k
    return audio_24k


def _ensure_batched(x: torch.Tensor) -> torch.Tensor:
    # (T, D) -> (1, T, D)
    if x.dim() == 2:
        return x.unsqueeze(0)
    return x


def generate_mel_sample(model, motion, lyrics, device):
    """
    生成用: z ~ N(0, I) をサンプルして decoder のみで mel を生成する。
    encoder(audio, ...) を通さないので audio 入力は不要。
    """
    model.eval()
    motion = _ensure_batched(motion).to(device)
    lyrics = _ensure_batched(lyrics).to(device)

    if motion.size(0) == 0:
        raise RuntimeError(f"Empty batch for motion: {tuple(motion.shape)}")

    B = motion.size(0)

    # latent_dim の取得（モデルに属性がなければ CONFIG から）
    latent_dim = getattr(model, "latent_dim", None) or CONFIG["latent_dim"]
    z = torch.randn(B, latent_dim, device=device)

    if not hasattr(model, "decoder"):
        raise AttributeError("Model has no attribute 'decoder'. Cannot sample without encoder.")

    # decoder の引数名に合わせて呼ぶ
    sig = inspect.signature(model.decoder.forward)
    params = sig.parameters

    kwargs = {}
    if "z" in params:
        kwargs["z"] = z
    else:
        # 位置引数しかない/名前が違うケースは fallback
        return model.decoder(z, motion, lyrics)

    if "motion" in params:
        kwargs["motion"] = motion
    if "lyrics" in params:
        kwargs["lyrics"] = lyrics

    # よくある別名にも対応
    if "x" in params and "motion" not in params:
        kwargs["x"] = motion
    if "y" in params and "lyrics" not in params:
        kwargs["y"] = lyrics

    mel = model.decoder.forward(**kwargs)
    return mel  # (B, T, n_mels) 想定


def generate_audio(*args, **kwargs):
    """
    旧: encoder 経由 (audio 必須) の経路。
    推論では使わない（shape崩れの原因になる）ので禁止。
    """
    raise RuntimeError(
        "generate_audio() is disabled for inference. "
        "Use generate_mel_sample() (decoder-only sampling) instead."
    )


def _print_mel_stats(name, x):
    x = x.detach().float().cpu()
    print(f"[{name}] shape={tuple(x.shape)} min={x.min().item():.3f} max={x.max().item():.3f} mean={x.mean().item():.3f} std={x.std().item():.3f}")


def denorm_like_gt(mel_norm: torch.Tensor, gt_norm: torch.Tensor) -> torch.Tensor:
    """
    mel_norm: (T, M) 生成物（正規化空間）
    gt_norm : (T, M) 入力バッチのGT特徴（同じ正規化のされ方をしている前提）
    生成物を「GTと同じ平均・分散」に戻して dB 近似にする
    """
    mu = gt_norm.mean()
    sig = gt_norm.std().clamp_min(1e-6)
    return mel_norm * sig + mu


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # モデルをロード
    model = TransCVAE(
        audio_dim=CONFIG["audio_dim"],
        motion_dim=CONFIG["motion_dim"],
        lyrics_dim=CONFIG["lyrics_dim"],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_layers=CONFIG["num_layers"],
        dim_feedforward=CONFIG["dim_feedforward"],
        latent_dim=CONFIG["latent_dim"],
        seq_len=CONFIG["seq_len"],
    ).to(device)
    
    checkpoint_path = "checkpoints/transcvae_final.pt"
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first by running: python train.py")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint: {checkpoint_path}")
    
    # デバッグ: decoder のシグネチャ確認（想定と違う場合の切り分け用）
    try:
        print("decoder.forward signature:", inspect.signature(model.decoder.forward))
    except Exception as e:
        print("Could not inspect decoder.forward:", e)

    loader = get_2022_dataloader(batch_size=1, max_songs=3)

    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)

    for idx, batch in enumerate(loader):
        motion = batch["motion"]
        lyrics = batch["lyrics"]
        song_name = batch["song"][0] if isinstance(batch["song"], (list, tuple)) else str(batch["song"])
        timestamp = batch["timestamp"][0] if isinstance(batch["timestamp"], (list, tuple)) else str(batch["timestamp"])

        # decoder-only sampling
        generated = generate_mel_sample(model, motion, lyrics, device)  # 形が (B*T, 128) の可能性あり

        mel_bt = _as_bt_mel(generated.detach().cpu(), seq_len=CONFIG["seq_len"], n_mels=128)  # (B, T, 128)
        mel_db = mel_bt[0]  # (T, 128) ← ここが重要（[0]は最後に）

        # 生成物を GT と同じスケールに戻す（dB 近似）
        audio = mel_to_audio(denorm_like_gt(mel_db, batch["audio"][0].cpu()), target_sr=CONFIG.get("sr", 18000))
        audio_np = audio.squeeze(0).detach().cpu().numpy()

        out_path = os.path.join(output_dir, f"{song_name}_{timestamp}.wav")
        sf.write(out_path, audio_np, CONFIG.get("sr", 18000))
        print(f"✓ Generated: {out_path}")

        # デバッグ用: 生成したメルスペクトログラムの統計情報を表示
        _print_mel_stats("generated_mel_db(?)", mel_db)

        # バッチに GT mel があるなら:
        # _print_mel_stats("gt_audio_feat", batch["audio"][0].cpu())

        if idx >= 4:
            break

    print(f"\n✓ Generation complete! Check {output_dir}/ for results.")

if __name__ == "__main__":
    main()