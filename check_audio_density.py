import librosa
import numpy as np
import soundfile as sf

# 元のファイルの分析
original_path = '/Users/yusuke.yano/python_practice/murmur_analysis/input/buminko2LSB.wav'
segment_path = '/Users/yusuke.yano/python_practice/murmur_analysis/output/buminko2LSB_segment_1.wav'

def analyze_audio(file_path):
    print(f"\n=== {file_path} の分析 ===")
    
    # librosaでの読み込み
    y_librosa, sr_librosa = librosa.load(file_path, sr=None)
    
    # soundfileでの読み込み（より詳細な情報を取得）
    y_sf, sr_sf = sf.read(file_path)
    info = sf.info(file_path)
    
    print(f"サンプリングレート: {sr_librosa} Hz")
    print(f"データ長: {len(y_librosa)} サンプル")
    print(f"時間長: {len(y_librosa)/sr_librosa:.3f} 秒")
    print(f"データ型: {y_librosa.dtype}")
    print(f"\nファイル情報:")
    print(f"フォーマット: {info.format}")
    print(f"サブタイプ: {info.subtype}")
    print(f"チャンネル数: {info.channels}")
    
    # STFTのパラメータを計算
    n_fft = 2048  # STFTの窓サイズ
    hop_length = 512  # STFTのホップ長
    
    # STFTを計算
    D = librosa.stft(y_librosa, n_fft=n_fft, hop_length=hop_length)
    
    print(f"\nSTFT分析:")
    print(f"時間フレーム数: {D.shape[1]}")
    print(f"周波数ビン数: {D.shape[0]}")
    print(f"時間分解能: {hop_length/sr_librosa*1000:.2f} ミリ秒")
    print(f"周波数分解能: {sr_librosa/n_fft:.2f} Hz")

print("=== 音声データの密度分析 ===")
analyze_audio(original_path)
analyze_audio(segment_path) 