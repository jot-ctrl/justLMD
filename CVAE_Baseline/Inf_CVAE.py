from multiprocessing import freeze_support
import librosa
import torch
import torch.utils.data
from transformers import BertTokenizer, BertModel
import numpy as np

import os
from tqdm import tqdm

# cuda setup
if torch.cuda.is_available():
    device = torch.device("cuda")
else: device = torch.device("cpu")

kwargs = {'num_workers': 2, 'pin_memory': True} 

# hyper params
batch_size = 1
epochs = 10

feature_size = 500 * 72 * batch_size
latent_size = 512 * 512
class_size = (768*100 + 128*200) * batch_size  # 409600

fps = 25
sr = 16000
        
max_audio_length = 600
max_dance_length = 600
max_lyrics_length = 50

if os.path.exists('/home/yiyu/'):
    path = '/home/yiyu/JustLM2D/'
else: path = '/Users/Marvin/NII_Code/JustLM2D/'

from CVAE import *
    
def inference(model, music, lyrics):
    model.eval()
    with torch.no_grad():
        music, lyrics = music.to(device), lyrics.to(device)
        conditions = torch.concat((torch.flatten(music),torch.flatten(lyrics)),dim=0)
        sample = torch.randn(latent_size).to(device)
        out = model.decode(sample, conditions)
        print(out.shape)
        torch.save(out, 'inf.pt')

if __name__ == '__main__':
    freeze_support()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
        
    # LYRICS
    lyrics = "Oh, she's sweet but a psycho. A little bit psycho. At night she's screamin'. \"I'm-ma-ma-ma out my mind\""
    tokens = tokenizer.encode_plus(lyrics, add_special_tokens=True, return_tensors='pt')
    outputs = model(**tokens)
    lyrics_embeddings = outputs.last_hidden_state[0].T.detach().type(torch.FloatTensor)
    # [22, 768] to [100, 768]
    lyrics_embeddings = torch.nn.functional.pad(lyrics_embeddings, pad=(0, max_lyrics_length - lyrics_embeddings.size(1)), mode='constant', value=0)
    # lyrics_embeddings = torch.nn.functional.pad(lyrics_embeddings, pad=(0, max_lyrics_length - lyrics_embeddings.size(0)), mode='constant', value=0)
    
    # AUDIO
    audio,sr = librosa.load('/Users/Marvin/NII_Code/JustLM2D/Test/SweetButPsychoAvaMaxJustDance2023Edition/audios/48.wav', sr=sr)
    
    # Extract features (e.g. Mel spectrogram)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db_norm = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)
    # Convert to PyTorch tensor
    audio_feat = torch.from_numpy(mel_spec_db_norm).type(torch.FloatTensor)
    audio_feat = torch.nn.functional.pad(audio_feat, pad=(0, max_audio_length - audio_feat.size(1) ), mode='constant', value=0)

    # load a CVAE model
    model = CVAE(feature_size, latent_size, class_size).to(device)
    model.state_dict(torch.load(path+'Baseline/Model.pth'))
    # model = torch.load(path+'Baseline/Model.pth').to(device)

    # INFERENCE
    inference(model, audio_feat, lyrics_embeddings)
