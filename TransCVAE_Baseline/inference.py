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
    あなたの data_loader.py の mel 定義（SR=18000, hop_length=601, n_mels=128）に合わせて復元する。
    まずは vocoder 不整合を排除して「モデルが意味のある mel を出しているか」を確実に切り分ける。
    """
    if mel_db.dim() != 2:
        raise RuntimeError(f"mel_to_audio expects (T, n_mels), got {tuple(mel_db.shape)}")

    mel_np = mel_db.detach().cpu().numpy().T  # (n_mels, T)
    mel_power = librosa.db_to_power(mel_np)

    # librosa の逆メル。n_fft は melspectrogram のデフォルト(2048)に合わせる
    y = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=target_sr,
        n_fft=2048,
        hop_length=601,
        power=2.0,
    )
    # torch tensor で返す（既存の save 処理と合わせる）
    return torch.from_numpy(y).unsqueeze(0).float()

def _ensure_batched(x: torch.Tensor) -> torch.Tensor:
    # (T, D) -> (1, T, D)
    if x.dim() == 2:
        return x.unsqueeze(0)
    return x

def reconstruct_mel(model, audio, motion, lyrics, device):
    model.eval()
    audio  = _ensure_batched(audio).to(device)
    motion = _ensure_batched(motion).to(device)
    lyrics = _ensure_batched(lyrics).to(device)
    with torch.no_grad():
        recon, mu, logvar = model(audio, motion, lyrics)
    return recon.squeeze(0).cpu()  # (T, 128)

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
        audio_gt = batch["audio"]
        motion = batch["motion"]
        lyrics = batch["lyrics"]
        song_name = batch["song"][0] if isinstance(batch["song"], (list, tuple)) else str(batch["song"])
        timestamp = batch["timestamp"][0] if isinstance(batch["timestamp"], (list, tuple)) else str(batch["timestamp"])

        # ★再構成（encoder→decoder）
        recon_bt = reconstruct_mel(model, audio_gt, motion, lyrics, device)  # (T,128)想定 or (B,T,128) の可能性
        recon_bt = _as_bt_mel(recon_bt, seq_len=CONFIG["seq_len"], n_mels=128)  # (B,T,128)に揃える
        mel_db = recon_bt[0]  # (T,128)

        # 生成物を GT と同じスケールに戻す（dB 近似）
        audio = mel_to_audio(mel_db, target_sr=CONFIG.get("sr", 18000))
        audio_np = audio.squeeze(0).detach().cpu().numpy()

        out_path = os.path.join(output_dir, f"{song_name}_{timestamp}_RECON.wav")
        sf.write(out_path, audio_np, CONFIG.get("sr", 18000))
        print(f"✓ Generated: {out_path}")

        # デバッグ用: 生成したメルスペクトログラムの統計情報を表示
        _print_mel_stats("recon_mel_db(?)", mel_db)
        _print_mel_stats("gt_audio_feat", _as_bt_mel(audio_gt.detach().cpu(), seq_len=CONFIG["seq_len"], n_mels=128)[0])

        # バッチに GT mel があるなら:
        # _print_mel_stats("gt_audio_feat", batch["audio"][0].cpu())
        
        # --- 追加: GT mel を同じ逆変換で wav化して、逆変換の妥当性を確認 ---
        gt_bt = _as_bt_mel(audio_gt.detach().cpu(), seq_len=CONFIG["seq_len"], n_mels=128)
        gt_mel = gt_bt[0]  # (T, 128)

        gt_audio = mel_to_audio(gt_mel, target_sr=CONFIG.get("sr", 18000))
        gt_audio_np = gt_audio.squeeze(0).detach().cpu().numpy()

        gt_out_path = os.path.join(output_dir, f"{song_name}_{timestamp}_GTINV.wav")
        sf.write(gt_out_path, gt_audio_np, CONFIG.get("sr", 18000))
        print(f"✓ Wrote GT inversion: {gt_out_path}")

        break

    print(f"\n✓ Generation complete! Check {output_dir}/ for results.")

if __name__ == "__main__":
    main()