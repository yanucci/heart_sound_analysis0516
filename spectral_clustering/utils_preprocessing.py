import numpy as np
import librosa
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d

def bandpass_filter(data, sr, lowcut=20, highcut=5000, order=4):
    nyq = 0.5 * sr
    # highcutがナイキスト周波数を超えないようにする
    highcut = min(highcut, nyq - 1)
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def normalize_audio(audio):
    # 振幅を-1〜1に正規化
    return audio / np.max(np.abs(audio))

def minmax_scale_audio(audio):
    # 振幅を0〜1にスケーリング
    return (audio - np.min(audio)) / (np.max(audio) - np.min(audio) + 1e-8)

def rescale_duration(audio, target_length=512):
    # 信号長を0-1区間に線形リスケーリング
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, audio, kind='linear')
    return f(x_new)

def extract_stft(audio, sr, n_fft=256, hop_length=64):
    n_fft = min(n_fft, len(audio))
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    return mag

def extract_mfcc(audio, sr, n_mfcc=13):
    n_fft = min(256, len(audio))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    return mfcc.T 