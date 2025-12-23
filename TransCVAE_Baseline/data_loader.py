"""
Songs_Test の2つのデータのみで学習するためのデータセット実装。
Motion (SMPL) + Lyrics (BERT) → Audio (mel-spectrogram) の CVAE 用。
"""

import os
import json
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import BertTokenizer, BertModel

# グローバル設定
FPS = 30
SR = 18000
SEQUENCE_LENGTH = 6  # 秒


def to_seconds(time_stamp):
    minutes, seconds = map(float, time_stamp.split(':'))
    return datetime.timedelta(minutes=minutes, seconds=seconds).total_seconds()


def to_timestamp(seconds):
    delta = datetime.timedelta(seconds=seconds)
    return '{:02d}:{:06.3f}'.format(int(delta.total_seconds() // 60), delta.total_seconds() % 60)


class SongsTestDataset(Dataset):
    """
    Songs_Test ディレクトリから motion, lyrics, audio を読み込む。
    """
    def __init__(self):
        # 修正: 実際のプロジェクトルートに合わせる
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.test_dir = os.path.join(repo_root, 'Songs_Test')  # self.test_dir に保存
        
        if not os.path.exists(self.test_dir):
            raise FileNotFoundError(f"Songs_Test directory not found at: {self.test_dir}")
        
        self.songs = sorted([d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))])
        
        # BERT tokenizer and model for lyrics embedding
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()

        self.data = []
        self._load_data()
    
    def _load_data(self):
        """各曲のシーケンスをロード"""
        for song_name in self.songs:
            song_path = os.path.join(self.test_dir, song_name)
            
            # 必須ファイル確認
            if not os.path.exists(os.path.join(song_path, 'sliced.json')):
                continue
            if not os.path.exists(os.path.join(song_path, 'output-smpl-3d/smplfull.json')):
                continue
            if not os.path.exists(os.path.join(song_path, 'audio.wav')):
                continue
            if not os.path.exists(os.path.join(song_path, 'lyrics.lrc')):
                continue
            
            # sliced.json をロード（歌詞タイムスタンプ）
            with open(os.path.join(song_path, 'sliced.json'), 'r') as f:
                sliced = json.load(f)
            
            # SMPL データをロード
            with open(os.path.join(song_path, 'output-smpl-3d/smplfull.json'), 'r') as f:
                full_dance = json.load(f)
            
            start_sec = to_seconds(list(sliced.keys())[0])
            timestamps = list(sliced.keys())[:-1]  # 最後のシーケンスは除外
            
            for timestamp in timestamps:
                timestamp_sec = to_seconds(timestamp)
                trimmed_timestamp = to_timestamp(timestamp_sec - start_sec)
                tag = str(int(timestamp_sec))
                
                # Motion (SMPL) をロード
                motion_data = self._load_dance(full_dance, trimmed_timestamp)
                if motion_data is None:
                    continue
                
                # Audio (mel-spectrogram) をロード
                audio_data = self._load_audio(
                    os.path.join(song_path, 'audio.wav'),
                    timestamp_sec
                )
                if audio_data is None:
                    continue
                
                # Lyrics をロード
                lyrics_text = sliced[timestamp]
                lyrics_data = self._load_lyrics(lyrics_text)
                if lyrics_data is None:
                    continue
                
                self.data.append({
                    'motion': motion_data,
                    'lyrics': lyrics_data,
                    'audio': audio_data,
                    'song': song_name,
                    'timestamp': tag
                })
    
    def _load_dance(self, full_dance, timestamp):
        """SMPL データをロード (FPS=30, SEQUENCE_LENGTH=6秒) - 正規化付き"""
        try:
            dance = []
            start_frame = to_seconds(timestamp) * FPS
            all_frames = list(full_dance.keys())
            
            for offset in range(SEQUENCE_LENGTH * FPS):
                frame_idx = str(int(start_frame + offset)).zfill(6)
                if frame_idx not in all_frames:
                    return None
                
                frame_data = full_dance[frame_idx]
                if 'annots' not in frame_data or len(frame_data['annots']) == 0:
                    return None
                
                annot = frame_data['annots'][0]
                
                # poses, Th, Rh を取得（リストまたは ndarray を想定）
                poses = annot.get('poses', [[]])[0]
                Th = annot.get('Th', [[]])[0]
                Rh = annot.get('Rh', [[]])[0]
                
                # リストを ndarray に変換
                if isinstance(poses, list):
                    poses = np.array(poses, dtype=np.float32)
                else:
                    poses = np.asarray(poses, dtype=np.float32)
                
                if isinstance(Th, list):
                    Th = np.array(Th, dtype=np.float32)
                else:
                    Th = np.asarray(Th, dtype=np.float32)
                
                if isinstance(Rh, list):
                    Rh = np.array(Rh, dtype=np.float32)
                else:
                    Rh = np.asarray(Rh, dtype=np.float32)
                
                # 結合
                combined = np.concatenate([poses, Th, Rh])
                dance.append(combined)
            
            if len(dance) != SEQUENCE_LENGTH * FPS:
                return None
            
            # torch.tensor() で直接変換（from_numpy の ABI 問題を回避）
            dance_np = np.stack(dance, axis=0).astype(np.float32)
            dance = torch.tensor(dance_np, dtype=torch.float32)
            
            # 正規化
            dance_mean = dance.mean(dim=0, keepdim=True)
            dance_std = dance.std(dim=0, keepdim=True) + 1e-8
            dance = (dance - dance_mean) / dance_std
            
            return dance
        except Exception as e:
            print(f"Error loading dance: {e}")
            return None
    
    def _load_audio(self, audio_path, start_sec):
        """Mel-spectrogram をロード"""
        try:
            audio, _ = librosa.load(audio_path, sr=SR, offset=start_sec, duration=SEQUENCE_LENGTH)
            
            # Mel-spectrogram 抽出
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SR, hop_length=601, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            audio_features = torch.from_numpy(mel_spec_db.T).float()
            return audio_features
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
    
    def _load_lyrics(self, lyrics_text):
        """BERT で歌詞を埋め込み"""
        try:
            with torch.no_grad():
                tokens = self.tokenizer.encode_plus(
                    lyrics_text,
                    add_special_tokens=True,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding=True,
                )
                outputs = self.bert_model(**tokens)
                # 最後のレイヤーの出力を使用
                lyrics_embeddings = outputs.last_hidden_state[0].detach()  # (seq_len, 768)
                
                # mean pooling -> (1, 768)
                lyrics_mean = lyrics_embeddings.mean(dim=0, keepdim=True)

                # repeat to (180, 768)
                lyrics_seq = lyrics_mean.expand(SEQUENCE_LENGTH * FPS, -1).contiguous()
            return lyrics_seq.float()
        except Exception as e:
            print(f"Error loading lyrics: {e}")
            return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'motion': item['motion'],      # (180, motion_dim)
            'lyrics': item['lyrics'],      # (180, 128)
            'audio': item['audio'],        # (time_steps, 128)
            'song': item['song'],
            'timestamp': item['timestamp']
        }


def get_test_dataloader(batch_size=4, num_workers=0):
    """Songs_Test 用 DataLoader を生成"""
    dataset = SongsTestDataset()
    print(f"Loaded {len(dataset)} sequences from Songs_Test")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader


# TransCVAE_Baseline/data_loader.py に Songs_2022 用データセット追加
class Songs2022Dataset(Dataset):
    """
    Songs_2022 ディレクトリから motion, lyrics, audio を読み込む。
    """
    def __init__(self, songs_dir=None, max_songs=3, max_samples=None, max_timestamps_per_song=None):
        # 修正: 実際のプロジェクトルートに合わせる
        if songs_dir is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            songs_dir = os.path.join(repo_root, 'Songs_2022')
        
        self.songs_dir = songs_dir
        all_songs = sorted([d for d in os.listdir(songs_dir) if os.path.isdir(os.path.join(songs_dir, d))])
        self.songs = all_songs[:max_songs] if max_songs else all_songs  # max_songs=None で全部

        # 追加: スモークテスト/デバッグ用のロード上限
        self.max_samples = max_samples
        self.max_timestamps_per_song = max_timestamps_per_song
        
        # BERT tokenizer and model for lyrics embedding
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """各曲のシーケンスをロード"""
        for song_name in self.songs:
            song_path = os.path.join(self.songs_dir, song_name)
            
            # 必須ファイル確認
            if not os.path.exists(os.path.join(song_path, 'sliced.json')):
                continue
            if not os.path.exists(os.path.join(song_path, 'output-smpl-3d/smplfull.json')):
                continue
            if not os.path.exists(os.path.join(song_path, 'lyrics.lrc')):
                continue
            if not os.path.exists(os.path.join(song_path, 'audio.wav')):
                print(f"[Warning] audio.wav not found in {song_path}, skipping")
                continue
            
            # sliced.json をロード
            with open(os.path.join(song_path, 'sliced.json'), 'r') as f:
                sliced = json.load(f)
            
            # SMPL データをロード
            with open(os.path.join(song_path, 'output-smpl-3d/smplfull.json'), 'r') as f:
                full_dance = json.load(f)
            
            start_sec = to_seconds(list(sliced.keys())[0])
            timestamps = list(sliced.keys())[:-1]

            # 追加: 1曲あたりのタイムスタンプ上限（BERT計算が重いので）
            if self.max_timestamps_per_song is not None:
                timestamps = timestamps[: int(self.max_timestamps_per_song)]
            
            for timestamp in timestamps:
                timestamp_sec = to_seconds(timestamp)
                trimmed_timestamp = to_timestamp(timestamp_sec - start_sec)
                tag = str(int(timestamp_sec))
                
                # Motion (SMPL) をロード
                motion_data = self._load_dance(full_dance, trimmed_timestamp)
                if motion_data is None:
                    continue
                
                # Audio (mel-spectrogram) をロード
                audio_data = self._load_audio(
                    os.path.join(song_path, 'audio.wav'),
                    timestamp_sec
                )
                if audio_data is None:
                    continue
                
                # Lyrics をロード
                lyrics_text = sliced[timestamp]
                lyrics_data = self._load_lyrics(lyrics_text)
                if lyrics_data is None:
                    continue
                
                self.data.append({
                    'motion': motion_data,
                    'lyrics': lyrics_data,
                    'audio': audio_data,
                    'song': song_name,
                    'timestamp': tag
                })

                # 追加: サンプル総数の上限
                if self.max_samples is not None and len(self.data) >= int(self.max_samples):
                    return
    
    def _load_dance(self, full_dance, timestamp):
        """SMPL データをロード (FPS=30, SEQUENCE_LENGTH=6秒) - 正規化付き"""
        try:
            dance = []
            start_frame = to_seconds(timestamp) * FPS
            all_frames = list(full_dance.keys())
            
            for offset in range(SEQUENCE_LENGTH * FPS):
                frame_idx = str(int(start_frame + offset)).zfill(6)
                if frame_idx not in all_frames:
                    return None
                
                frame_data = full_dance[frame_idx]
                if 'annots' not in frame_data or len(frame_data['annots']) == 0:
                    return None
                
                annot = frame_data['annots'][0]
                
                # poses, Th, Rh を取得（リストまたは ndarray を想定）
                poses = annot.get('poses', [[]])[0]
                Th = annot.get('Th', [[]])[0]
                Rh = annot.get('Rh', [[]])[0]
                
                # リストを ndarray に変換
                if isinstance(poses, list):
                    poses = np.array(poses, dtype=np.float32)
                else:
                    poses = np.asarray(poses, dtype=np.float32)
                
                if isinstance(Th, list):
                    Th = np.array(Th, dtype=np.float32)
                else:
                    Th = np.asarray(Th, dtype=np.float32)
                
                if isinstance(Rh, list):
                    Rh = np.array(Rh, dtype=np.float32)
                else:
                    Rh = np.asarray(Rh, dtype=np.float32)
                
                # 結合
                combined = np.concatenate([poses, Th, Rh])
                dance.append(combined)
            
            if len(dance) != SEQUENCE_LENGTH * FPS:
                return None
            
            # torch.tensor() で直接変換（from_numpy の ABI 問題を回避）
            dance_np = np.stack(dance, axis=0).astype(np.float32)
            dance = torch.tensor(dance_np, dtype=torch.float32)
            
            # 正規化
            dance_mean = dance.mean(dim=0, keepdim=True)
            dance_std = dance.std(dim=0, keepdim=True) + 1e-8
            dance = (dance - dance_mean) / dance_std
            
            return dance
        except Exception as e:
            print(f"Error loading dance: {e}")
            return None
    
    def _load_audio(self, audio_path, start_sec):
        """Mel-spectrogram をロード - 固定長対応"""
        try:
            audio, _ = librosa.load(audio_path, sr=SR, offset=start_sec, duration=SEQUENCE_LENGTH)
            
            # Mel-spectrogram 抽出
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=SR, hop_length=601, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            audio_features = torch.from_numpy(mel_spec_db.T).float()
            
            # **固定長にリサンプル (SEQUENCE_LENGTH * FPS = 180)**
            target_length = SEQUENCE_LENGTH * FPS
            if audio_features.shape[0] != target_length:
                # 線形補間でリサンプル
                audio_features = torch.nn.functional.interpolate(
                    audio_features.unsqueeze(0).permute(0, 2, 1),
                    size=target_length,
                    mode='linear',
                    align_corners=False
                ).squeeze(0).permute(1, 0)
            
            return audio_features
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
    
    def _load_lyrics(self, lyrics_text):
        """BERT で歌詞を埋め込み - 固定サイズ投影"""
        try:
            with torch.no_grad():
                tokens = self.tokenizer.encode_plus(
                    lyrics_text,
                    add_special_tokens=True,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True
                )
                outputs = self.bert_model(**tokens)
                lyrics_embeddings = outputs.last_hidden_state[0].detach()  # (seq_len, 768)
                
                # **pool: 平均をとってから投影（より安定）**
                lyrics_mean = lyrics_embeddings.mean(dim=0, keepdim=True)  # (1, 768)
                
                # フレーム数に合わせて拡張
                lyrics_embeddings = lyrics_mean.expand(SEQUENCE_LENGTH * FPS, -1)  # (180, 768)

            return lyrics_embeddings.float()
        except Exception as e:
            print(f"Error loading lyrics: {e}")
            return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'motion': item['motion'],      # (180, motion_dim) - 正規化済み
            'lyrics': item['lyrics'],      # (180, 128)
            'audio': item['audio'],        # (180, 128) - 固定長
            'song': item['song'],
            'timestamp': item['timestamp']
        }


def get_2022_dataloader(batch_size=1, max_songs=3, max_samples=None, max_timestamps_per_song=None):
    """Songs_2022 用 DataLoader を生成"""
    dataset = Songs2022Dataset(
        max_songs=max_songs,
        max_samples=max_samples,
        max_timestamps_per_song=max_timestamps_per_song,
    )
    print(f"Loaded {len(dataset)} sequences from Songs_2022 ({dataset.songs[:max_songs]})")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader


def get_multiseason_dataloader(batch_size=4, years=['2020', '2021', '2022']):
    """
    複数シーズン (Songs_2020, Songs_2021, Songs_2022) を結合したデータローダーを返す。
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    all_data = []
    for year in years:
        songs_dir = os.path.join(repo_root, f'Songs_{year}')
        if not os.path.exists(songs_dir):
            print(f"[Warning] {songs_dir} not found, skipping")
            continue
        
        dataset = Songs2022Dataset(songs_dir=songs_dir, max_songs=None)  # max_songs=None で全部読む
        all_data.extend(dataset.data)
    
    if not all_data:
        raise ValueError("No data loaded from any year")
    
    print(f"[DataLoader] Loaded {len(all_data)} samples from years {years}")
    
    return DataLoader(
        all_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )


def get_season_dataloader(batch_size=4, year='2022', max_songs=None, max_samples=None, max_timestamps_per_song=None):
    """
    特定シーズン (Songs_YYYY) のデータローダーを返す。
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    songs_dir = os.path.join(repo_root, f'Songs_{year}')
    
    if not os.path.exists(songs_dir):
        raise FileNotFoundError(f"Songs_{year} directory not found at: {songs_dir}")
    
    dataset = Songs2022Dataset(
        songs_dir=songs_dir,
        max_songs=max_songs,
        max_samples=max_samples,
        max_timestamps_per_song=max_timestamps_per_song,
    )
    
    return DataLoader(
        dataset.data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )


def collate_batch(batch):
    """
    バッチ内のサンプルを辞書形式でまとめる。
    各サンプルは {'motion': ..., 'lyrics': ..., 'audio': ...} の形。
    """
    motions = []
    lyrics = []
    audios = []
    
    for sample in batch:
        motions.append(sample['motion'])
        lyrics.append(sample['lyrics'])
        audios.append(sample['audio'])
    
    # テンソルにスタック（可変長の場合はpadding処理が必要）
    motions = torch.stack(motions, dim=0)
    lyrics = torch.stack(lyrics, dim=0)
    
    # オーディオは可変長の可能性があるため、最大長でパディング
    max_audio_len = max(a.shape[0] for a in audios)
    audios_padded = []
    for audio in audios:
        if audio.shape[0] < max_audio_len:
            pad = max_audio_len - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, 0, 0, pad))
        audios_padded.append(audio)
    
    audios = torch.stack(audios_padded, dim=0)
    
    return {
        'motion': motions,
        'lyrics': lyrics,
        'audio': audios,
    }


if __name__ == "__main__":
    # テスト
    loader = get_test_dataloader(batch_size=2)
    for batch in loader:
        print("Motion shape:", batch['motion'].shape)
        print("Lyrics shape:", batch['lyrics'].shape)
        print("Audio shape:", batch['audio'].shape)
        print("Songs:", batch['song'])
        break

    # Songs_2022 データセットテスト
    loader_2022 = get_2022_dataloader(batch_size=1, max_songs=3)
    for batch in loader_2022:
        print("Motion shape:", batch['motion'].shape)
        print("Lyrics shape:", batch['lyrics'].shape)
        print("Songs:", batch['song'])
        break

    # 複数シーズンデータローダーテスト
    multi_loader = get_multiseason_dataloader(batch_size=2, years=['2020', '2021', '2022'])
    for batch in multi_loader:
        print("Motion shape:", batch['motion'].shape)
        print("Lyrics shape:", batch['lyrics'].shape)
        print("Songs:", batch['song'])
        break

    # 特定シーズンデータローダーテスト
    season_loader = get_season_dataloader(batch_size=2, year='2022', max_songs=3)
    for batch in season_loader:
        print("Motion shape:", batch['motion'].shape)
        print("Lyrics shape:", batch['lyrics'].shape)
        print("Songs:", batch['song'])
        break
