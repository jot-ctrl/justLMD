# TransCVAE_Baseline

Motion+Lyrics を条件に Audio を生成する CVAE のベースライン実装です。

- 条件: Motion (pose/features), Lyrics (text embedding)
- 生成: Audio feature (e.g., mel-spectrogram frames)

## 構成
- `src/models/architectures/transcvae.py`: モデル本体 (Conditional VAE)
- `train.py`: 学習スクリプト (ダミーのデータローダー雛形付き)
- `config.py`: ハイパーパラメータ定義

## 使い方
```bash
# 依存関係 (例)
python -m venv .venv
source .venv/bin/activate
pip install torch torchaudio numpy librosa transfomers

# 学習 (ダミーデータで形状確認)
python train.py
```

本ベースラインは形状と学習ループの雛形を提供します。実運用では、
- Motion 特徴量の前処理 (例: SMPL/rotation2xyz 由来のベクトル)
- Lyrics 埋め込み (例: Transformer/BERT など)
- Audio 特徴量 (例: MelSpectrogram) と復元ロス (L1/MSE/SSIM など)
- 実データの DataLoader
を接続してください。