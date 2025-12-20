from GLOBAL import *
import re

todo = []
for song in getSongList(version):
    if song == 'urls' or song.startswith('.') or song.startswith('_'):
        continue

    song_path = songs_dir + song
    if not os.path.isdir(song_path):
        continue

    videos_path = os.path.join(song_path, 'videos')
    annots_path = os.path.join(song_path, 'annots')
    output_path = os.path.join(song_path, 'output-smpl-3d')

    if not os.path.isdir(videos_path) or len(os.listdir(videos_path)) == 0:
        continue

    annots_exists = os.path.isdir(annots_path)
    if os.path.exists(output_path) and (not annots_exists or len(os.listdir(annots_path)) == 0):
        continue

    todo.append(song)

# todo.sort()
print(todo)
# todo = todo[::-1]

for song in todo:
    song_path = songs_dir + song
    smplmesh_path = os.path.join(song_path, 'output-smpl-3d', 'smplmesh')
    if os.path.isdir(smplmesh_path):
        for dir in os.listdir(smplmesh_path):
            os.system('rm -rf %s/images/%s' % (song_path, dir))
            os.system('rm -rf %s/annots/%s' % (song_path, dir))

os.chdir('./.EasyMocap/')

for song in todo:
    song_path = songs_dir + song

    os.system('python apps/demo/mocap.py %s --work internet --fps 30 --bodyonly' % song_path) 
    #--disable_vismesh \
    
    smplmesh_path = os.path.join(song_path, 'output-smpl-3d', 'smplmesh')
    if os.path.isdir(smplmesh_path):
        for dir in os.listdir(smplmesh_path):
            os.system('rm -rf %s/images/%s' % (song_path, dir))
            os.system('rm -rf %s/annots/%s' % (song_path, dir))
            
for song in os.listdir(songs_dir):
    song_path = songs_dir + song
    smpl_path = os.path.join(song_path, 'output-smpl-3d')
    smplmesh_path = os.path.join(smpl_path, 'smplmesh')
    if os.path.isdir(smplmesh_path):
        for dir in os.listdir(smplmesh_path):
            if dir.endswith('.mp4'):
                toDel = dir[:-4]
                os.system('rm -rf %s/output-smpl-3d/smplmesh/%s' % (song_path, toDel))
    images_path = os.path.join(song_path, 'images')
    annots_path = os.path.join(song_path, 'annots')
    cache_spin_path = os.path.join(song_path, 'cache_spin')
    if os.path.isdir(images_path) and os.path.isdir(annots_path) and os.path.isdir(cache_spin_path) \
        and len(os.listdir(images_path)) == 0 and len(os.listdir(annots_path)) == 0:
        os.system('rm -rf %s/images' % song_path)
        os.system('rm -rf %s/annots' % song_path)
        os.system('rm -rf %s/cache_spin' % song_path)
        
for song in os.listdir(songs_dir):
    if song.startswith('.') or song.startswith('_'):
        continue

    song_path = songs_dir + song
    os.system('rm -rf %s/cache_spin' % song_path)

    smplfull_video_path = os.path.join(song_path, 'output-smpl-3d', 'smplfull', 'video')
    if not os.path.isdir(smplfull_video_path):
        continue

    all_frames = {}
    for frame in os.listdir(smplfull_video_path):
        if frame.endswith('.json'):
            json_path = os.path.join(smplfull_video_path, frame)
            all_frames[frame[:-5]] = json.load(open(json_path, 'r'))

    if not all_frames:
        continue

    smplfull_json_path = os.path.join(song_path, 'output-smpl-3d', 'smplfull.json')
    json.dump(all_frames, open(smplfull_json_path, 'w', encoding="utf-8"), ensure_ascii=False)

    with open(smplfull_json_path, 'r') as f:
        text = f.read()

    # Remove newlines after "[" character using regular expressions
    text = re.sub(r'(".*?":)', r'\n\1', text)
    text = re.sub(r'("K"|"R"|"T"|"annots")', r'\t\1', text)
    text = re.sub(r'("id"|"shapes"|"poses"|"Rh"|"Th")', r'\t\t\1', text)

    # Open the file for writing and write the modified text
    with open(smplfull_json_path, 'w') as f:
        f.write(text)
    
    print(song)