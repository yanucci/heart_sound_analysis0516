import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog,
                           QSpinBox, QListWidget)
from PyQt6.QtCore import Qt
import pandas as pd
import soundfile as sf

class AudioSegmentVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('音声データ・セグメント可視化')
        self.setGeometry(100, 100, 1600, 1000)
        
        # データ保持用の変数
        self.audio = None
        self.sr = None
        self.peaks_data = None
        self.segments = []  # セグメントデータを保持
        self.window_size = 5  # 表示する時間幅（秒）
        self.current_position = 0  # 現在の表示位置（秒）
        self.current_segment_index = -1  # 現在選択中のセグメント
        
        # UIの初期化
        self.init_ui()
        
    def init_ui(self):
        # メインウィジェット
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 左側のコントロールパネル
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # ファイル操作ボタン
        self.load_audio_button = QPushButton('音声ファイルを開く')
        self.load_audio_button.clicked.connect(self.load_audio)
        left_layout.addWidget(self.load_audio_button)
        
        self.load_csv_button = QPushButton('ピークデータを開く')
        self.load_csv_button.clicked.connect(self.load_peaks)
        left_layout.addWidget(self.load_csv_button)
        
        # セグメント作成ボタン
        self.create_segments_button = QPushButton('セグメントを作成')
        self.create_segments_button.clicked.connect(self.create_segments)
        self.create_segments_button.setEnabled(False)
        left_layout.addWidget(self.create_segments_button)
        
        # セグメント保存ボタン
        self.save_segments_button = QPushButton('セグメントを保存')
        self.save_segments_button.clicked.connect(self.save_segments)
        self.save_segments_button.setEnabled(False)
        left_layout.addWidget(self.save_segments_button)
        
        # セグメントリスト
        self.segment_list = QListWidget()
        self.segment_list.itemSelectionChanged.connect(self.on_segment_selected)
        left_layout.addWidget(QLabel('セグメントリスト:'))
        left_layout.addWidget(self.segment_list)
        
        main_layout.addWidget(left_panel, stretch=1)
        
        # 右側のグラフ表示部分
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # グラフ表示用のmatplotlibキャンバス
        self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        
        # ツールバーの追加
        self.toolbar = NavigationToolbar(self.canvas, right_panel)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
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
        right_layout.addLayout(position_layout)
        
        # 窓サイズ調整
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel('窓サイズ(秒):'))
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setRange(1, 60)
        self.window_spinbox.setValue(self.window_size)
        self.window_spinbox.valueChanged.connect(self.update_window_size)
        window_layout.addWidget(self.window_spinbox)
        right_layout.addLayout(window_layout)
        
        main_layout.addWidget(right_panel, stretch=4)
    
    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '音声ファイルを選択', '', 'WAV files (*.wav)')
        if file_path:
            # 音声データの読み込み
            self.audio, self.sr = librosa.load(file_path, sr=None)
            self.audio_path = file_path
            self.position_slider.setMaximum(int(len(self.audio) / self.sr) - self.window_size)
            self.update_plot()
    
    def load_peaks(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'ピークデータを選択', '', 'CSV files (*.csv)')
        if file_path:
            # ピークデータの読み込み
            self.peaks_data = pd.read_csv(file_path)
            self.create_segments_button.setEnabled(True)
            self.update_plot()
    
    def create_segments(self):
        if self.audio is None or self.peaks_data is None:
            return
            
        # セグメントをクリア
        self.segments.clear()
        self.segment_list.clear()
        
        # ピークの時間でソート
        sorted_peaks = self.peaks_data.sort_values('time')
        peak_times = sorted_peaks['time'].values
        
        # セグメントの作成
        for i in range(len(peak_times) - 1):
            start_time = peak_times[i]
            end_time = peak_times[i + 1]
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            
            segment = {
                'start_time': start_time,
                'end_time': end_time,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'audio_data': self.audio[start_sample:end_sample],
                'label': f'Segment {i+1} ({start_time:.2f}s - {end_time:.2f}s)'
            }
            self.segments.append(segment)
            self.segment_list.addItem(segment['label'])
        
        self.save_segments_button.setEnabled(True)
        self.update_plot()
    
    def save_segments(self):
        if not self.segments:
            return
            
        # 保存ディレクトリの選択
        save_dir = QFileDialog.getExistingDirectory(self, 'セグメント保存先を選択')
        if not save_dir:
            return
            
        # 元のファイル名を取得
        base_filename = os.path.splitext(os.path.basename(self.audio_path))[0]
        
        # 各セグメントを保存
        for i, segment in enumerate(self.segments):
            # WAVファイルとして保存
            filename = f'{base_filename}_segment_{i+1}.wav'
            filepath = os.path.join(save_dir, filename)
            sf.write(filepath, segment['audio_data'], self.sr)
        
        print(f'セグメントを保存しました: {save_dir}')
    
    def on_segment_selected(self):
        selected_items = self.segment_list.selectedItems()
        if not selected_items:
            self.current_segment_index = -1
        else:
            self.current_segment_index = self.segment_list.row(selected_items[0])
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
        self.ax3.clear()
        
        # 全体波形の表示（ax1）
        time_full = np.arange(len(self.audio)) / self.sr
        self.ax1.plot(time_full, self.audio)
        self.ax1.set_title('全体波形')
        self.ax1.set_xlabel('時間 (秒)')
        self.ax1.set_ylabel('振幅')
        
        # ピークの表示
        if self.peaks_data is not None:
            peak_times = self.peaks_data['time'].values
            peak_amplitudes = self.peaks_data['amplitude'].values
            self.ax1.plot(peak_times, peak_amplitudes, 'rx')
        
        # 現在の表示範囲を強調表示
        self.ax1.axvspan(self.current_position, 
                        self.current_position + self.window_size, 
                        color='yellow', alpha=0.3)
        
        # 拡大表示部分（ax2）
        start_sample = int(self.current_position * self.sr)
        end_sample = int((self.current_position + self.window_size) * self.sr)
        
        time = np.arange(start_sample, end_sample) / self.sr
        self.ax2.plot(time, self.audio[start_sample:end_sample])
        
        # 表示範囲内のピークを表示
        if self.peaks_data is not None:
            visible_peaks = self.peaks_data[
                (self.peaks_data['time'] >= self.current_position) & 
                (self.peaks_data['time'] < self.current_position + self.window_size)
            ]
            
            if not visible_peaks.empty:
                self.ax2.plot(visible_peaks['time'], visible_peaks['amplitude'], 'rx')
                
                for _, peak in visible_peaks.iterrows():
                    self.ax2.annotate(peak['peak_number'], 
                                    (peak['time'], peak['amplitude']),
                                    xytext=(0, 10), textcoords='offset points', 
                                    ha='center')
        
        self.ax2.set_title('拡大表示')
        self.ax2.set_xlabel('時間 (秒)')
        self.ax2.set_ylabel('振幅')
        
        # 選択中のセグメント表示（ax3）
        if self.current_segment_index >= 0 and self.segments:
            segment = self.segments[self.current_segment_index]
            time_segment = np.arange(len(segment['audio_data'])) / self.sr + segment['start_time']
            self.ax3.plot(time_segment, segment['audio_data'])
            self.ax3.set_title(f'選択中のセグメント: {segment["label"]}')
            self.ax3.set_xlabel('時間 (秒)')
            self.ax3.set_ylabel('振幅')
        else:
            self.ax3.set_title('セグメント未選択')
        
        plt.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioSegmentVisualizer()
    window.show()
    sys.exit(app.exec()) 