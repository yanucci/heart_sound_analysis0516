"""
心音セグメントからの特徴量抽出
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

def load_segment_data(segment_dir):
    """
    セグメントデータの読み込みと前処理
    """
    data = []
    labels = []
    
    # セグメントファイルの読み込み
    for file in tqdm(os.listdir(segment_dir)):
        if not file.endswith('.wav'):
            continue
            
        # ファイル名からデータセット名を抽出
        dataset_name = file.split('_segment_')[0]
        file_path = os.path.join(segment_dir, file)
        
        # 音声データの読み込み
        audio, sr = librosa.load(file_path, sr=None)
        
        data.append({
            'file_name': file,
            'dataset_name': dataset_name,
            'audio': audio,
            'sr': sr
        })
        labels.append(dataset_name)
    
    return data, labels

def extract_features(audio_data, sr):
    """
    音声データからの特徴量抽出
    """
    features = {}
    
    # 時間領域の特徴量
    features['rms'] = np.sqrt(np.mean(audio_data**2))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
    
    # スペクトル特徴量
    spec = np.abs(librosa.stft(audio_data))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=spec))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=spec))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=spec))
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_var'] = np.var(mfccs[i])
    
    # クロマグラム
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)
    
    # テンポラル特徴量
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
    features['tempo'] = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    return features

def create_feature_dataset(segment_dir):
    """
    特徴量データセットの作成
    """
    # データの読み込み
    data, labels = load_segment_data(segment_dir)
    
    # 特徴量の抽出
    features_list = []
    for item in tqdm(data):
        features = extract_features(item['audio'], item['sr'])
        features['file_name'] = item['file_name']
        features['dataset_name'] = item['dataset_name']
        features_list.append(features)
    
    # DataFrameの作成
    df = pd.DataFrame(features_list)
    
    return df

if __name__ == "__main__":
    segment_dir = "/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments"
    df = create_feature_dataset(segment_dir)
    
    # 特徴量の保存
    output_dir = os.path.dirname(segment_dir)
    df.to_csv(os.path.join(output_dir, "heart_sound_features.csv"), index=False)
    print("特徴量の抽出が完了しました。") 