import librosa


def get_beat_times(music_file):
    y, sr = librosa.load(music_file)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times