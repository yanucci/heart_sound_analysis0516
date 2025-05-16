"""
データ処理ユーティリティ
心音データの処理と解析を担当
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import os

class HeartSoundProcessor:
    def __init__(self, file_manager):
        """
        初期化
        Args:
            file_manager (HeartSoundFileManager): ファイル管理インスタンス
        """
        self.file_manager = file_manager
    
    def load_audio(self, name):
        """
        音声データの読み込み
        Args:
            name (str): データセット名
        Returns:
            tuple: (audio_data, sampling_rate)
        """
        info = self.file_manager.get_dataset_info(name)
        if info is None:
            raise ValueError(f"Dataset {name} not found")
            
        # 処理状態を'processing'に更新
        self.file_manager.update_status(name, 'processing')
            
        file_path = os.path.join(self.file_manager.input_dir, f"{name}.wav")
        return librosa.load(file_path, sr=None)
    
    def save_peaks(self, name, peaks_data):
        """
        ピーク情報の保存
        Args:
            name (str): データセット名
            peaks_data (dict): ピーク情報
        """
        df = pd.DataFrame(peaks_data)
        output_path = self.file_manager.get_peak_log_path(name)
        df.to_csv(output_path, index=False)
        self.file_manager.update_status(name, 'peaks_detected')
    
    def create_segments(self, name, audio_data, sr, peaks):
        """
        セグメント作成
        Args:
            name (str): データセット名
            audio_data (np.ndarray): 音声データ
            sr (int): サンプリングレート
            peaks (list): ピーク位置のリスト
        """
        for i in range(len(peaks) - 1):
            start_sample = int(peaks[i] * sr)
            end_sample = int(peaks[i + 1] * sr)
            segment = audio_data[start_sample:end_sample]
            
            # セグメントの保存
            output_path = self.file_manager.get_segment_path(name, i + 1)
            sf.write(output_path, segment, sr, subtype='PCM_16')
        
        # 処理完了を記録
        self.file_manager.update_status(name, 'completed')
    
    def analyze_segment(self, audio_data, sr):
        """
        セグメントの解析
        Args:
            audio_data (np.ndarray): 音声データ
            sr (int): サンプリングレート
        Returns:
            dict: 解析結果
        """
        # 基本的な特徴量の計算
        duration = len(audio_data) / sr
        rms = np.sqrt(np.mean(audio_data**2))
        
        # スペクトル特徴量
        D = librosa.stft(audio_data)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # 主要な周波数成分
        freqs = librosa.fft_frequencies(sr=sr)
        main_freq_idx = np.argmax(np.mean(np.abs(D), axis=1))
        main_freq = freqs[main_freq_idx]
        
        return {
            'duration': duration,
            'rms': rms,
            'main_frequency': main_freq,
            'max_amplitude': np.max(np.abs(audio_data)),
            'mean_db': np.mean(D_db)
        }
    
    def get_processing_info(self, name):
        """
        処理状態の情報を取得
        Args:
            name (str): データセット名
        Returns:
            dict: 処理状態の情報
        """
        return self.file_manager.get_processing_status(name) 