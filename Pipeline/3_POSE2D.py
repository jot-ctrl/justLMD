from GLOBAL import *

fps = 30
sr = 16000
    
if __name__ == '__main__':
    todo = []
    for song in getSongList(version):
        if song == 'urls' or song.startswith('.') or song.startswith('_'):
            continue

        song_path = songs_dir + song
        if not os.path.isdir(song_path):
            continue

        videos_path = '%s/videos' % song_path
        if not os.path.isdir(videos_path):
            continue

        annots_path = '%s/annots' % song_path
        if len(os.listdir(videos_path)) == 0 \
            or ( os.path.exists('%s/output-smpl-3d'%(song_path)) and (not os.path.exists(annots_path) or len(os.listdir(annots_path)) == 0)):
            continue
        todo.append(song)

    # todo.sort()
    # todo = todo[::-1]
    print(todo)

    os.chdir('./.EasyMocap/')
    for song in todo:
        song_path = songs_dir + song
        os.system('python apps/preprocess/extract_keypoints.py %s --mode mp-pose'%song_path)