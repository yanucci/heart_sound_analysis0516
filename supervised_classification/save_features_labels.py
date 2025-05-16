import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# supervised_classification_heart_sound.pyと同じ特徴量・ラベル抽出処理
features = []
labels = []

import os
import librosa
from utils_preprocessing import bandpass_filter, normalize_audio, minmax_scale_audio, rescale_duration, extract_mfcc

data_dir = '/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments'
csv_path = '/Users/yusuke.yano/python_practice/heart_sound_analysis/data/dataset_index.csv'
wav_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]

df_index = pd.read_csv(csv_path)
name_set = set(df_index['name'].values)

for wav_path in wav_files:
    fname = os.path.basename(wav_path)
    label = fname.split('_')[0]
    if label not in name_set:
        continue
    audio, sr = librosa.load(wav_path, sr=5000)
    audio = bandpass_filter(audio, sr)
    audio = normalize_audio(audio)
    audio = minmax_scale_audio(audio)
    audio = rescale_duration(audio, target_length=512)
    mfcc = extract_mfcc(audio, sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=0)
    features.append(mfcc_mean)
    labels.append(label)
features = np.array(features)
labels = np.array(labels)

np.save('features.npy', features)
np.save('labels.npy', labels)
print('features.npy, labels.npy を保存しました') 