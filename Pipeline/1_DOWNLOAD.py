from GLOBAL import *
import argparse

def getURLs(version):
    pipeline_dir = os.path.join(repo_root, 'Pipeline')
    os.makedirs(pipeline_dir, exist_ok=True)
    file_path = os.path.join(pipeline_dir, version + ".json")
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"urls": []}, f, ensure_ascii=False, indent=4)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data.get("urls", [])
            if isinstance(data, list):
                return data
            return []
    except json.JSONDecodeError:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"urls": []}, f, ensure_ascii=False, indent=4)
        return []

def urlToTitle(version):
    urls = getURLs(version)
    songList = getSongList(version)
    finished = list(songList.values())
    
    if len(urls) == len(finished):
        print("Json already done")
        return
    
    pipeline_dir = os.path.join(repo_root, 'Pipeline')
    os.makedirs(pipeline_dir, exist_ok=True)
    file_path = os.path.join(pipeline_dir, version + ".json")
    
    for u in urls:
        # urls entries can be plain string or [url, crop]
        url_str = u[0] if isinstance(u, (list, tuple)) and len(u) > 0 else (u if isinstance(u, str) else None)
        if url_str is None:
            continue
        # skip if already present (match by url string)
        if any((isinstance(v, (list, tuple)) and len(v) > 0 and v[0] == url_str) or (isinstance(v, str) and v == url_str) for v in finished):
            continue
        # get title with yt-dlp
        title = os.popen('yt-dlp --get-title %s' % url_str).read().strip()
        print(title)
        print(u)
        songList[re.sub(r'\W+', '', title)] = u
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"urls": urls, **songList}, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and process JustDance videos')
    parser.add_argument('--max-songs', type=int, default=None, 
                        help='Maximum number of songs to process (default: all)')
    args = parser.parse_args()
    
    urlToTitle(version)
    songList = getSongList(version)
    
    processed_count = 0
    
    for song in songList.keys():
        # skip the metadata key 'urls' if present in songList dict
        if song == "urls":
            continue
        
        if args.max_songs is not None and processed_count >= args.max_songs:
            print(f"\n[Stopped after {args.max_songs} songs]")
            break

        song_path = os.path.join(songs_dir, re.sub(r'\W+', '', song))
        os.makedirs(song_path, exist_ok=True)
        
        entry = songList[song]
        # normalize entry -> url_str and crop
        if isinstance(entry, (list, tuple)):
            url_str = entry[0] if len(entry) > 0 else None
        elif isinstance(entry, dict):
            url_str = entry.get("url") or entry.get("urls") or None
        elif isinstance(entry, str):
            url_str = entry
        else:
            url_str = None
        
        if not url_str:
            print(f"Skipping {song}: no url found")
            continue

        if not os.path.exists(os.path.join(song_path, "video.mp4")):
            print(f"downloading: {song}")
            # 以前の失敗したダウンロードファイルや一時ファイルを削除
            os.system('rm -rf "%s"/*.mp4 "%s"/*.part "%s"/*.ytdl' % (song_path, song_path, song_path))
            
            # yt-dlpコマンドの構築
            # --extractor-args "youtube:player_client=android": Webクライアント制限(403エラー)を回避する強力なオプション
            # --socket-timeout 30: 接続タイムアウトを設定
            download_cmd = (
                'yt-dlp '
                '--rm-cache-dir '
                '--no-check-certificate '
                '--force-ipv4 '
                '--extractor-args "youtube:player_client=android" '
                '-f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" '
                '--merge-output-format mp4 '
                '--socket-timeout 30 '
                '-o "%s" '
                '"%s"'
            ) % (os.path.join(song_path, "video.mp4"), url_str)
            
            result = os.system(download_cmd)
            
            if result != 0:
                print(f"WARNING: Download failed for {song}. Trying fallback method...")
                # フォールバック: iOSクライアントとして試行
                fallback_cmd = (
                    'yt-dlp '
                    '--no-check-certificate '
                    '--extractor-args "youtube:player_client=ios" '
                    '-f "best" '
                    '-o "%s" '
                    '"%s"'
                ) % (os.path.join(song_path, "video.mp4"), url_str)
                os.system(fallback_cmd)
        
        processed_count += 1
        
    # 音声変換処理
    print("\n=== Converting videos to audio ===")
    audio_processed = 0
    audio_count = 0
    for song in songList.keys():
        if song == "urls":
            continue
        if args.max_songs is not None and audio_count >= args.max_songs:
            break
        song_path = os.path.join(songs_dir, re.sub(r'\W+', '', song))
        audio_count += 1
        
        if not os.path.exists(os.path.join(song_path, "audio.wav")):
            if os.path.exists(os.path.join(song_path, "video.mp4")):
                print(f"Converting to audio: {song}")
                result = os.system('ffmpeg -i %s/video.mp4 -ab 160k -ac 2 -ar %s -vn %s/audio.wav -y 2>&1 | grep -E "(Duration|time=)" | tail -1' % (song_path, str(sr), song_path))
                if result == 0:
                    audio_processed += 1
            else:
                print(f"Skipping {song}: video.mp4 not found")
        else:
            print(f"Audio already exists: {song}")
    
    print(f"\n[Processing complete. Videos: {processed_count}, Audio conversions: {audio_processed}]")
