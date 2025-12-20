from multiprocessing import freeze_support
import torch
import torch.utils.data
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import os
import sys
from tqdm import tqdm

# start tensorboard 

# cuda setup
if torch.cuda.is_available():
    device = torch.device("cuda")
else: device = torch.device("cpu")

kwargs = {'num_workers': 2, 'pin_memory': True} 

# hyper params
batch_size = 1
epochs = 100

feature_size = 600 * 72 * batch_size
latent_size = 512 * 512
class_size = (768*50 + 128*600) * batch_size  # 


current_dir = os.path.abspath(os.path.dirname(__file__))

def _find_repo_root(start_dir):
    d = start_dir
    for _ in range(8):
        if os.path.exists(os.path.join(d, '.git')) and os.path.exists(os.path.join(d, 'Pipeline')):
            return d
        d = os.path.abspath(os.path.join(d, '..'))
    return start_dir

repo_root = _find_repo_root(current_dir)
pipeline_dir = os.path.join(repo_root, 'Pipeline')
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)

from LMD_Dataset import LMD_Dataset

def _songs_collection(root):
    years = ['2020', '2021', '2022']
    return [os.path.join(root, f'Songs_{year}') + os.sep for year in years]

data_dir = os.path.join(pipeline_dir)
songs_collection = _songs_collection(repo_root)
model_dir = current_dir

from CVAE import *

def train(epoch, model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, item in enumerate(train_loader):
        lyrics, music, dance = item['lyrics'].to(device), item['music'].to(device), item['dance'].to(device)
        # print("Lyrics:",lyrics.shape, "Music:" ,music.shape, "Dance:" ,dance.shape)
        # labels = torch.concat(one_hot(music, 10), one_hot(lyrics, 10))
        labels = torch.concat((torch.flatten(music),torch.flatten(lyrics)),dim=0)
        dance = torch.flatten(dance)
        recon_batch, mu, logvar = model(dance, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, dance, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(lyrics), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(lyrics)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch, model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            labels = one_hot(labels, 10)
            recon_batch, mu, logvar = model(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
            if i == 0:
                n = min(data.size(0), 5)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(-1, 1, 500, 72)[:n]])
                save_image(comparison.cpu(),
                         'reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    # print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == '__main__':
    freeze_support()

    dataset = LMD_Dataset(data_dir, songs_collection, name='LMD_New_Embedding')

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # create a CVAE model
    model = CVAE(feature_size, latent_size, class_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in tqdm(range(1, epochs + 1)):
        train(epoch, model, train_loader, optimizer)
        # test(epoch, model, test_loader)
        # with torch.no_grad():
        #     c = torch.eye(10, 10).to(device)
        #     sample = torch.randn(10, 20).to(device)
        #     sample = model.decode(sample, c).cpu()
        #     save_image(sample.view(10, 1, 500, 72),
        #                 'sample_' + str(epoch) + '.png')
    
    torch.save(model.state_dict(), os.path.join(model_dir, 'Model.pth'))
