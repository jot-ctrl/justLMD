import os
import json
import random

if os.path.exists('/home/yiyu/'):
    path = '/home/yiyu/JustLM2D/'
else: path = '/Users/Marvin/NII_Code/JustLM2D/'

import sys
sys.path.insert(0, path + '/Pipeline/')
sys.path.insert(0, path + '/TransVAE_Baseline/')
from LMD_Dataset import LMD_Dataset

import matplotlib.pyplot as plt
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.utils.tensors import collate72, collate78
from src.models.get_model import get_model as get_gen_model

from src.parser.training import parser

parameters = parser()
parameters['device'] = torch.device('cpu')

INF_MODE = 78 #78 bits: 24joints + global rotation + transition
# INF_MODE = 72 #72 bits: 24joints

exp_name = 'lm2d20230622-125926' #with glob and trans
# exp_name = 'lm2d20230624-105702' #with glob and trans
# exp_name = 'lm2d20230626-145127' #without glob and trans

test_path = path + '/Songs_Test/'
test_dataset = LMD_Dataset(path + '/Pipeline/', [test_path], name='Test')

model = get_gen_model(parameters)
state_dict = torch.load(path + "/TransVAE_Baseline/exps/%s/checkpoint_5000.pth.tar"%exp_name, map_location='cpu')
model.load_state_dict(state_dict)

seq_name = random.choice(list(test_dataset.indexing.values()))
# seq_name = 'AllTheStarsbyKendrickLamarftSZAJustDance2021_57'
# seq_name = 'FitButYouKnowItJustDance2020xTzShark_64'
test_sequence = test_dataset.LMD_Dict[seq_name]
batch = [test_sequence]

if INF_MODE == 72: batch = collate72(batch)
if INF_MODE == 78: batch = collate78(batch)
batch['z'] = torch.randn(1, 512)

# inference
model.eval()
batch = model.decoder(batch)
out = batch['output']
out = out[0]
out = out.permute(2,0,1)

if INF_MODE == 72: 
    out = out.reshape(180,72)
    out = torch.cat([out, torch.zeros((180,6))], 1)
if INF_MODE == 78: out = out.reshape(180,78)

[song, tag] = seq_name.split('_')
torch.save(out,'%s/%s.pt'%(test_path + song, seq_name))

# test_dataset.export(seq_name, save_dir='Previews/%s/%s_groundtruth'%(exp_name,seq_name), inf=False)
# test_dataset.export(seq_name, save_dir='Previews/%s/%s'%(exp_name,seq_name), inf=True, glob_trans=False)

if INF_MODE==78: test_dataset.export(seq_name, save_dir='Previews/%s/%s_glob_tans'%(exp_name,seq_name), inf=True)
if INF_MODE==72: test_dataset.export(seq_name, save_dir='Previews/%s/%s'%(exp_name,seq_name), inf=True, glob_trans=False)
