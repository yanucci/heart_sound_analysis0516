import sys
import numpy as np
import librosa
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog,
                           QCheckBox, QSpinBox)
from PyQt6.QtCore import Qt
import pandas as pd

class HeartSoundSegmentation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('音声セグメンテーション')
        self.setGeometry(100, 100, 1400, 800)
        
        # データ保持用の変数
        self.audio = None
        self.sr = None
        self.peaks = None
        self.threshold = 0.5
        self.window_size = 5  # 表示する時間幅（秒）
        self.current_position = 0  # 現在の表示位置（秒）
        self.selected_peak = None  # 選択中のピーク
        self.stft_matrix = None  # STFTデータを保持
        self.freqs = None  # 周波数ビンを保持
        
        # UIの初期化
        self.init_ui()
        
    def init_ui(self):
        # メインウィジェット
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # グラフ表示用のmatplotlibキャンバス
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # ツールバーの追加（ズームやパンなどの機能を提供）
        self.toolbar = NavigationToolbar(self.canvas, main_widget)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # コントロールパネル
        control_layout = QHBoxLayout()
        
        # ファイル選択ボタン
        self.load_button = QPushButton('音声ファイルを開く')
        self.load_button.clicked.connect(self.load_audio)
        control_layout.addWidget(self.load_button)
        
        # 窓サイズ調整
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel('窓サイズ(秒):'))
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setRange(1, 60)
        self.window_spinbox.setValue(self.window_size)
        self.window_spinbox.valueChanged.connect(self.update_window_size)
        window_layout.addWidget(self.window_spinbox)
        control_layout.addLayout(window_layout)
        
        # 表示位置調整用スライダー
        position_layout = QVBoxLayout()
        self.position_label = QLabel('表示位置: 0.00秒')
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(100)
        self.position_slider.setValue(0)
        self.position_slider.valueChanged.connect(self.update_position)
        position_layout.addWidget(self.position_label)
        position_layout.addWidget(self.position_slider)
        control_layout.addLayout(position_layout)
        
        # ピーク削除ボタン
        self.delete_button = QPushButton('選択したピークを削除')
        self.delete_button.clicked.connect(self.delete_selected_peak)
        self.delete_button.setEnabled(False)
        control_layout.addWidget(self.delete_button)
        
        # 保存ボタン
        self.save_button = QPushButton('セグメンテーション結果を保存')
        self.save_button.clicked.connect(self.save_segmentation)
        self.save_button.setEnabled(False)
        control_layout.addWidget(self.save_button)
        
        layout.addLayout(control_layout)
        
        # マウスイベントの設定
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '音声ファイルを選択', '', 'WAV files (*.wav)')
        if file_path:
            # 音声データの読み込み
            self.audio, self.sr = librosa.load(file_path, sr=None)
            self.file_path = file_path
            self.peaks = []  # ピークをリセット
            self.selected_peak = None
            
            # STFTの計算
            self.stft_matrix = librosa.stft(self.audio)
            self.freqs = librosa.fft_frequencies(sr=self.sr)
            
            self.position_slider.setMaximum(int(len(self.audio) / self.sr) - self.window_size)
            self.update_plot()
            self.save_button.setEnabled(True)
            self.delete_button.setEnabled(False)
    
    def update_window_size(self):
        self.window_size = self.window_spinbox.value()
        if self.audio is not None:
            self.position_slider.setMaximum(int(len(self.audio) / self.sr) - self.window_size)
            self.update_plot()
    
    def update_position(self):
        self.current_position = self.position_slider.value()
        self.position_label.setText(f'表示位置: {self.current_position:.2f}秒')
        if self.audio is not None:
            self.update_plot()
    
    def on_click(self, event):
        if event.inaxes != self.ax1 or self.audio is None:
            return
            
        clicked_time = event.xdata
        if clicked_time is None:
            return
            
        absolute_time = self.current_position + clicked_time
        sample_position = int(absolute_time * self.sr)
        
        if event.button == 1:  # 左クリック：ピーク追加
            self.peaks.append(sample_position)
            self.peaks.sort()
            self.selected_peak = None
            self.delete_button.setEnabled(False)
        elif event.button == 3:  # 右クリック：ピーク選択
            # 最も近いピークを探す
            if self.peaks:
                visible_peaks = [p for p in self.peaks if abs(p - sample_position) < self.sr]
                if visible_peaks:
                    nearest_peak = min(visible_peaks, key=lambda p: abs(p - sample_position))
                    self.selected_peak = nearest_peak
                    self.delete_button.setEnabled(True)
                else:
                    self.selected_peak = None
                    self.delete_button.setEnabled(False)
        
        self.update_plot()
    
    def delete_selected_peak(self):
        if self.selected_peak is not None and self.selected_peak in self.peaks:
            self.peaks.remove(self.selected_peak)
            self.selected_peak = None
            self.delete_button.setEnabled(False)
            self.update_plot()
    
    def update_plot(self):
        if self.audio is None:
            return
            
        # グラフのクリア
        self.ax1.clear()
        self.ax2.clear()
        
        # 表示範囲の設定
        start_sample = int(self.current_position * self.sr)
        end_sample = int((self.current_position + self.window_size) * self.sr)
        
        # 波形の表示
        time = np.arange(start_sample, end_sample) / self.sr - self.current_position
        self.ax1.plot(time, self.audio[start_sample:end_sample])
        
        # ピークの表示
        visible_peaks = [p for p in self.peaks if start_sample <= p < end_sample]
        if visible_peaks:
            peak_times = np.array(visible_peaks) / self.sr - self.current_position
            peak_amplitudes = self.audio[visible_peaks]
            
            # 通常のピーク
            self.ax1.plot(peak_times, peak_amplitudes, 'rx')
            
            # 選択中のピークを強調表示
            if self.selected_peak in visible_peaks:
                selected_time = self.selected_peak / self.sr - self.current_position
                selected_amplitude = self.audio[self.selected_peak]
                self.ax1.plot(selected_time, selected_amplitude, 'go', markersize=10)
            
            # ピーク番号の表示
            for i, (peak_time, peak_amp) in enumerate(zip(peak_times, peak_amplitudes)):
                self.ax1.annotate(f'P{i+1}', (peak_time, peak_amp), 
                                xytext=(0, 10), textcoords='offset points', ha='center')
        
        self.ax1.set_title('音声波形とピーク検出')
        self.ax1.set_xlabel('時間 (秒)')
        self.ax1.set_ylabel('振幅')
        
        # スペクトログラムの表示
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio[start_sample:end_sample])), ref=np.max)
        librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz', ax=self.ax2)
        self.ax2.set_ylim(0, 5000)  # 周波数範囲を0-5000Hzに固定
        self.ax2.set_title('スペクトログラム')
        self.ax2.set_xlabel('時間 (秒)')
        
        plt.tight_layout()
        self.canvas.draw()
    
    def get_peak_features(self, peak_sample):
        """ピークの特徴量を抽出"""
        # 時間情報
        peak_time = peak_sample / self.sr
        
        # 振幅情報
        peak_amplitude = self.audio[peak_sample]
        
        # 周波数特性の抽出
        # STFTの時間フレームインデックスを計算
        frame_index = int(peak_sample / len(self.audio) * self.stft_matrix.shape[1])
        frame_index = min(frame_index, self.stft_matrix.shape[1] - 1)
        
        # そのフレームのスペクトル
        spectrum = np.abs(self.stft_matrix[:, frame_index])
        
        # 主要な周波数成分を抽出（上位3つ）
        top_freq_indices = np.argsort(spectrum)[-3:]
        top_freqs = self.freqs[top_freq_indices]
        top_magnitudes = spectrum[top_freq_indices]
        
        return {
            'time': peak_time,
            'amplitude': peak_amplitude,
            'freq1': top_freqs[2],
            'freq1_magnitude': top_magnitudes[2],
            'freq2': top_freqs[1],
            'freq2_magnitude': top_magnitudes[1],
            'freq3': top_freqs[0],
            'freq3_magnitude': top_magnitudes[0]
        }

    def save_segmentation(self):
        if self.peaks is not None and len(self.peaks) > 0:
            # 各ピークの特徴量を収集
            peak_features = []
            for i, peak in enumerate(self.peaks):
                features = self.get_peak_features(peak)
                features['peak_number'] = f'P{i+1}'
                peak_features.append(features)
            
            # DataFrameに変換
            results = pd.DataFrame(peak_features)
            
            # 列の順序を整理
            columns = ['peak_number', 'time', 'amplitude', 
                      'freq1', 'freq1_magnitude',
                      'freq2', 'freq2_magnitude', 
                      'freq3', 'freq3_magnitude']
            results = results[columns]
            
            # ファイル名を生成（元の音声ファイルと同じ場所）
            save_path = self.file_path.replace('.wav', '_peaks_analysis.csv')
            results.to_csv(save_path, index=False)
            print(f'ピーク解析結果を保存しました: {save_path}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HeartSoundSegmentation()
    window.show()
    sys.exit(app.exec())