import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.models.architectures.transcvae import TransCVAE, kld_loss, recon_loss_l1
from data_loader import get_multiseason_dataloader, get_season_dataloader
from config import CONFIG


def train(epochs=10, use_multiseason=True, year='2022'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
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

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    
    # データローダー選択
    if use_multiseason:
        print("[Train] Using multi-season data (2020, 2021, 2022)")
        loader = get_multiseason_dataloader(batch_size=CONFIG["batch_size"], years=['2020', '2021', '2022'])
    else:
        print(f"[Train] Using single season data (Songs_{year})")
        loader = get_season_dataloader(batch_size=CONFIG["batch_size"], year=year, max_songs=None)

    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            # 辞書から取り出し
            motion = batch["motion"].to(device)   # (B, T, motion_dim)
            lyrics = batch["lyrics"].to(device)   # (B, T, lyrics_dim)
            audio  = batch["audio"].to(device)    # (B, T, audio_dim)

            seq_len = CONFIG.get("seq_len", 180)

            # 念のため長さを揃える（padding/trim）
            if audio.size(1) < seq_len:
                audio = torch.nn.functional.pad(audio, (0,0,0, seq_len-audio.size(1)))
            else:
                audio = audio[:, :seq_len, :]

            motion = motion[:, :seq_len, :]
            lyrics = lyrics[:, :seq_len, :]

            recon, mu, logvar = model(audio, motion, lyrics)     # recon: (B, T, audio_dim) にする（次の修正）
            loss_recon = recon_loss_l1(recon, audio)
            loss_kld   = kld_loss(mu, logvar)
            loss       = loss_recon + CONFIG["kld_weight"] * loss_kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"Epoch {epoch} | step={step} | recon={loss_recon.item():.4f} kld={loss_kld.item():.4f} total={loss.item():.4f}")
    
    # チェックポイント保存
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/transcvae_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': CONFIG,
    }, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    # 複数年度で学習する場合
    train(epochs=20, use_multiseason=True)
    
    # または特定年度のみ
    # train(epochs=20, use_multiseason=False, year='2022')
