"""
心音データのGrad-CAM可視化
スペクトログラムに対してCNNを適用し、判断の根拠を可視化
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from heart_sound_model import HeartSoundCNNWithGradCAM

class HeartSoundDataset(Dataset):
    def __init__(self, segment_dir):
        self.segment_dir = segment_dir
        self.files = [f for f in os.listdir(segment_dir) if f.endswith('.wav')]
        self.labels = [f.split('_segment_')[0] for f in self.files]
        
        # ラベルのエンコーディング
        self.le = LabelEncoder()
        self.labels_encoded = self.le.fit_transform(self.labels)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.segment_dir, self.files[idx])
        audio, sr = librosa.load(file_path, sr=None)
        duration = len(audio) / sr
        
        # スペクトログラムの計算（最大5000Hzまで）
        n_fft = 2048
        hop_length = 512
        spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
        
        # 5000Hz以上の周波数成分を除外
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        max_freq_idx = np.where(freq_bins <= 5000)[0][-1]
        spec_db = spec_db[:max_freq_idx+1]
        
        # サイズの正規化（64x64にリサイズ）
        if spec_db.shape[1] > 64:
            spec_db = spec_db[:, :64]
        elif spec_db.shape[1] < 64:
            pad_width = 64 - spec_db.shape[1]
            spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
        
        # 正規化
        spec_db = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db))
        
        # PyTorchのテンソルに変換
        spec_tensor = torch.FloatTensor(spec_db).unsqueeze(0)
        label_tensor = torch.LongTensor([self.labels_encoded[idx]])
        
        return spec_tensor, label_tensor, self.files[idx], duration, sr, audio

def plot_spectrogram_with_gradcam(audio, sr, heatmap, filename, output_path):
    """
    心音のスペクトログラムとGrad-CAMを生成
    """
    duration = len(audio) / sr
    
    # スペクトログラムの計算
    n_fft = 2048
    hop_length = 512
    
    plt.figure(figsize=(15, 5))
    
    # オリジナルのスペクトログラム
    plt.subplot(1, 2, 1)
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    
    # 5000Hz以下の周波数成分のみ表示
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    max_freq_idx = np.where(freq_bins <= 5000)[0][-1]
    spec_db = spec_db[:max_freq_idx+1]
    
    librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='hz',
        cmap='magma',
        vmin=-60,
        vmax=0
    )
    
    plt.ylim(0, 5000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    # Grad-CAMヒートマップ
    plt.subplot(1, 2, 2)
    librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='hz',
        cmap='magma',
        vmin=-60,
        vmax=0
    )
    
    # ヒートマップのリサイズ
    heatmap_resized = np.repeat(np.repeat(heatmap, 8, axis=0), 8, axis=1)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5, aspect='auto', extent=[0, duration, 0, 5000])
    
    plt.ylim(0, 5000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Grad-CAM Heatmap')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    plt.suptitle(f'Heart Sound Analysis: {filename} (Duration: {duration:.2f}s)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_gradcam(model, spec, target_class, device):
    """
    Grad-CAMの生成
    """
    model.eval()
    spec = spec.requires_grad_(True)
    
    # 予測
    pred = model(spec.to(device))
    
    # 予測クラスに対する勾配を計算
    pred[:, target_class].backward()
    
    # 勾配を取得
    gradients = model.get_activations_gradient()
    
    # 活性化を取得
    activations = model.get_activations()
    
    # 重みの計算
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # 活性化に重みを掛ける
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    # ヒートマップの生成
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    
    return heatmap.detach().cpu().numpy()

def main():
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # データセットの準備
    segment_dir = "/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments"
    dataset = HeartSoundDataset(segment_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # モデルの準備
    model = HeartSoundCNNWithGradCAM(num_classes=len(dataset.le.classes_))
    model = model.to(device)
    
    # 学習済みモデルの読み込み
    state_dict = torch.load('results/best_model.pth')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Grad-CAMの生成と可視化
    output_dir = "results/gradcam"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Grad-CAMを生成中...")
    for spec, label, filename, duration, sr, audio in tqdm(dataloader):
        # Grad-CAMの計算
        heatmap = generate_gradcam(model, spec, label.item(), device)
        
        # スペクトログラムとGrad-CAMの可視化
        output_file = os.path.join(output_dir, f'gradcam_{filename[0]}.png')
        plot_spectrogram_with_gradcam(
            audio.squeeze().numpy(),
            sr.item(),
            heatmap,
            filename[0],
            output_file
        )
    
    print(f"\nGrad-CAM可視化結果を {output_dir} に保存しました。")

if __name__ == "__main__":
    main() 