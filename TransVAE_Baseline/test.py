import jukemirlib

audio = jukemirlib.load_audio('/home/yiyu/JustLM2D/Songs_2020/badguyBillieEilishJustDance2020/audio.wav')

reps = jukemirlib.extract(audio, layers=[36])

