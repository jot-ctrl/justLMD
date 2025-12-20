import datetime
# from torch.utils.data import Dataset
from .dataset import Dataset
import os

if os.path.exists('/home/yiyu/JustLM2D/'):
    home_path = '/home/yiyu/JustLM2D/'
else:
    home_path = '/Users/Marvin/NII_Code/JustLM2D/'

def toSeconds(time_stamp):
    minutes, seconds = map(float, time_stamp.split(':'))
    return datetime.timedelta(minutes=minutes, seconds=seconds).total_seconds()

#Lyrics_Music_Dance_Dataset
class LMD_Dataset(Dataset): 
    def __init__ (self, dict, indexing):
        self.LAD_Dict = dict
        self.indexing = indexing
        
    def __getitem__(self,index):
        key = self.indexing[str(index)]
        item = self.LAD_Dict[key]
        return item#['lyrics'], item['music'], item['dance']
    
    def __len__ (self):
        return len(self.indexing.keys())