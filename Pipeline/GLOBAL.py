import os
import json
import re
import datetime
import sys

# 既存の repo_root を確実にリポジトリルートにする（Pipeline の親フォルダ）
current_dir = os.path.abspath(os.path.dirname(__file__))
def _find_repo_root(start_dir):
    d = start_dir
    for _ in range(8):
        # adjust checks if your repo root has a known marker (e.g. .git or README.md)
        if os.path.exists(os.path.join(d, '.git')) or os.path.exists(os.path.join(d, 'Pipeline')):
            return d
        d = os.path.abspath(os.path.join(d, '..'))
    return start_dir

repo_root = _find_repo_root(current_dir)
sys.path.insert(0, os.path.join(repo_root, 'Pipeline'))

# if os.path.exists('/home/yiyu/'):
#     path = '/home/yiyu/JustLM2D/'
# else: path = '/Users/Marvin/NII_Code/JustLM2D/'
# Use repository root as default path so scripts run on any machine
path = repo_root + os.sep

# jd2022 = json.load(open(path + "/Pipeline/jd2022.json", "r"))
test = {'SweetButPsychoAvaMaxJustDance2023Edition':[]}

all_years = ['2020', '2021', '2022']
songs_collection = [path + 'Songs_2020/', path + 'Songs_2021/', path + 'Songs_2022/']

year = '2020'
songs_dir = path + 'Songs_'+year+'/'

fps = 30
# sr = 16000
sr = 18000
sequenceLength = 6
lyrics_padding = 180

version = 'JD'+year

def toSeconds(time_stamp):
    minutes, seconds = map(float, time_stamp.split(':'))
    return datetime.timedelta(minutes=minutes, seconds=seconds).total_seconds()

def toTimestamp(seconds): #format muniute:second.milisecond
    delta = datetime.timedelta(seconds=seconds)
    return '{:02d}:{:06.3f}'.format(int(delta.total_seconds() // 60), delta.total_seconds() % 60)

def getSongList(version):
    file_path = os.path.join(path, 'Pipeline', version + ".json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
    with open(file_path, "r", encoding="utf-8") as f:
        songList = json.load(f)
    return songList

def urlToTitle(version):
    urls = getURLs(version)
    songList = getSongList(version)
    finished = songList.values()
    for url in urls:
        if url not in finished:
            print(url)
    pipeline_file = os.path.join(repo_root, 'Pipeline', version + ".json")
    with open(pipeline_file, "w", encoding="utf-8") as f:
        json.dump(songList, f, ensure_ascii=False, indent=4)
