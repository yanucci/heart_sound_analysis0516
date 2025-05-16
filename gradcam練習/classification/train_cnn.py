"""
心音データのCNNモデル学習
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from gradcam_visualization import HeartSoundDataset
from heart_sound_model import HeartSoundCNN

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    モデルの学習
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        for specs, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            specs, labels = specs.to(device), labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for specs, labels, _ in val_loader:
                specs, labels = specs.to(device), labels.squeeze().to(device)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        accuracy = 100 * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {accuracy:.2f}%')
        
        # モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'results/best_model.pth')
    
    return train_losses, val_losses

def plot_training_history(train_losses, val_losses):
    """
    学習履歴の可視化
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_history.png')
    plt.close()

def main():
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # データセットの準備
    segment_dir = "/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments"
    dataset = HeartSoundDataset(segment_dir)
    
    # データセットの分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # データローダーの準備
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # モデルの準備
    model = HeartSoundCNN(num_classes=len(dataset.le.classes_))
    model = model.to(device)
    
    # 損失関数とオプティマイザの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 出力ディレクトリの作成
    os.makedirs('results', exist_ok=True)
    
    # モデルの学習
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, num_epochs=50,
        device=device
    )
    
    # 学習履歴の可視化
    plot_training_history(train_losses, val_losses)
    
    print("学習が完了しました。")
    print("モデルは'results/best_model.pth'に保存されました。")
    print("学習履歴は'results/training_history.png'に保存されました。")

if __name__ == "__main__":
    main() 