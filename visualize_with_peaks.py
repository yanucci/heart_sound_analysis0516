import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog,
                           QSpinBox)
from PyQt6.QtCore import Qt
import pandas as pd

class AudioVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('音声データ・ピーク可視化')
        self.setGeometry(100, 100, 1400, 800)
        
        # データ保持用の変数
        self.audio = None
        self.sr = None
        self.peaks_data = None
        self.window_size = 5  # 表示する時間幅（秒）
        self.current_position = 0  # 現在の表示位置（秒）
        
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
        
        # ツールバーの追加
        self.toolbar = NavigationToolbar(self.canvas, main_widget)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # コントロールパネル
        control_layout = QHBoxLayout()
        
        # 音声ファイル選択ボタン
        self.load_audio_button = QPushButton('音声ファイルを開く')
        self.load_audio_button.clicked.connect(self.load_audio)
        control_layout.addWidget(self.load_audio_button)
        
        # CSVファイル選択ボタン
        self.load_csv_button = QPushButton('ピークデータを開く')
        self.load_csv_button.clicked.connect(self.load_peaks)
        control_layout.addWidget(self.load_csv_button)
        
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
        
        layout.addLayout(control_layout)
    
    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '音声ファイルを選択', '', 'WAV files (*.wav)')
        if file_path:
            # 音声データの読み込み
            self.audio, self.sr = librosa.load(file_path, sr=None)
            self.position_slider.setMaximum(int(len(self.audio) / self.sr) - self.window_size)
            self.update_plot()
    
    def load_peaks(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'ピークデータを選択', '', 'CSV files (*.csv)')
        if file_path:
            # ピークデータの読み込み
            self.peaks_data = pd.read_csv(file_path)
            self.update_plot()
    
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
        if self.peaks_data is not None:
            # 表示範囲内のピークを抽出
            visible_peaks = self.peaks_data[
                (self.peaks_data['time'] >= self.current_position) & 
                (self.peaks_data['time'] < self.current_position + self.window_size)
            ]
            
            if not visible_peaks.empty:
                peak_times = visible_peaks['time'] - self.current_position
                peak_amplitudes = visible_peaks['amplitude']
                
                # ピークを表示
                self.ax1.plot(peak_times, peak_amplitudes, 'rx')
                
                # ピーク番号の表示
                for i, (time, amp, peak_num) in enumerate(zip(peak_times, peak_amplitudes, visible_peaks['peak_number'])):
                    self.ax1.annotate(peak_num, (time, amp), 
                                    xytext=(0, 10), textcoords='offset points', ha='center')
        
        self.ax1.set_title('音声波形とピーク位置')
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioVisualizer()
    window.show()
    sys.exit(app.exec()) 