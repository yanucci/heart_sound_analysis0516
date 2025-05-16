"""
心音セグメント可視化プログラム（安定版）
Version: 1.1.0

機能：
- 心音データのセグメントごとの波形表示
- 高品質なスペクトログラム表示
- STFTパラメータのリアルタイム調整
- 周波数範囲の動的変更
- 複数ファイルの一括読み込みと切り替え表示

改善点：
- グリッドスペックによる安定したレイアウト管理
- カラーバーの固定配置
- クリック時のレイアウト崩れ防止
- 日本語フォントの適切な処理

作成日: 2024-03-14
更新日: 2024-03-19
"""

import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel,
                           QSpinBox, QGroupBox, QListWidget,
                           QFileDialog)
from PyQt6.QtCore import Qt
import soundfile as sf

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'

class SegmentViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('セグメント波形確認')
        self.setGeometry(100, 100, 1400, 800)
        
        # データ変数
        self.audio_files = []  # ファイルパスのリスト
        self.current_index = -1  # 現在表示中のファイルインデックス
        self.audio_data = None
        self.sr = None
        
        # STFTパラメータ
        self.n_fft = 2048
        self.hop_length = 128
        self.win_length = 1024
        self.freq_min = 0
        self.freq_max = 5000
        
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
        
        # ファイル選択部分
        file_group = QGroupBox("ファイル操作")
        file_layout = QVBoxLayout()
        
        # ファイル選択ボタン
        load_button = QPushButton('ファイルを選択')
        load_button.clicked.connect(self.load_audio_files)
        file_layout.addWidget(load_button)
        
        # ファイルリスト
        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self.on_file_selected)
        file_layout.addWidget(self.file_list)
        
        # ファイル切り替えボタン
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton('前へ')
        self.prev_button.clicked.connect(self.show_previous)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton('次へ')
        self.next_button.clicked.connect(self.show_next)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)
        
        file_layout.addLayout(nav_layout)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # STFTパラメータ設定
        param_group = QGroupBox("STFTパラメータ")
        param_layout = QVBoxLayout()
        
        # FFTサイズ
        fft_layout = QHBoxLayout()
        fft_layout.addWidget(QLabel('FFTサイズ:'))
        self.fft_size = QSpinBox()
        self.fft_size.setRange(256, 4096)
        self.fft_size.setValue(self.n_fft)
        self.fft_size.valueChanged.connect(self.update_stft_params)
        fft_layout.addWidget(self.fft_size)
        param_layout.addLayout(fft_layout)
        
        # ホップ長
        hop_layout = QHBoxLayout()
        hop_layout.addWidget(QLabel('ホップ長:'))
        self.hop_size = QSpinBox()
        self.hop_size.setRange(64, 1024)
        self.hop_size.setValue(self.hop_length)
        self.hop_size.valueChanged.connect(self.update_stft_params)
        hop_layout.addWidget(self.hop_size)
        param_layout.addLayout(hop_layout)
        
        # 窓長
        win_layout = QHBoxLayout()
        win_layout.addWidget(QLabel('窓長:'))
        self.win_size = QSpinBox()
        self.win_size.setRange(256, 4096)
        self.win_size.setValue(self.win_length)
        self.win_size.valueChanged.connect(self.update_stft_params)
        win_layout.addWidget(self.win_size)
        param_layout.addLayout(win_layout)
        
        # 周波数範囲
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel('最大周波数(Hz):'))
        self.freq_max_spin = QSpinBox()
        self.freq_max_spin.setRange(1000, 24000)
        self.freq_max_spin.setValue(self.freq_max)
        self.freq_max_spin.valueChanged.connect(self.update_freq_range)
        freq_layout.addWidget(self.freq_max_spin)
        param_layout.addLayout(freq_layout)
        
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)
        
        # 情報表示
        info_group = QGroupBox("ファイル情報")
        info_layout = QVBoxLayout()
        self.info_label = QLabel('ファイル未選択')
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel, stretch=1)
        
        # 右側のグラフ表示部分
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # グラフ表示部分
        self.figure = plt.figure(figsize=(12, 8))
        gs = self.figure.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1])
        
        # 波形とスペクトログラムのサブプロット
        self.ax_wave = self.figure.add_subplot(gs[0, 0])    # 波形
        self.ax_spec = self.figure.add_subplot(gs[1, 0])    # スペクトログラム
        self.cax = self.figure.add_subplot(gs[1, 1])        # カラーバー
        
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, right_panel)
        
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        main_layout.addWidget(right_panel, stretch=3)
    
    def load_audio_files(self):
        """複数の音声ファイルを読み込む"""
        # ベースディレクトリの設定
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        initial_dir = os.path.join(current_dir, 'data', 'segments')
        
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            'セグメントファイルを選択',
            initial_dir,
            'WAV files (*.wav)'
        )
        
        if file_paths:
            # ファイルリストをクリアして新しいファイルを追加
            self.file_list.clear()
            self.audio_files = file_paths
            
            # リストウィジェットにファイル名を追加
            for file_path in file_paths:
                self.file_list.addItem(os.path.basename(file_path))
            
            # 最初のファイルを選択
            self.file_list.setCurrentRow(0)
            self.update_navigation_buttons()
    
    def on_file_selected(self):
        """ファイルリストで選択が変更されたときの処理"""
        current_row = self.file_list.currentRow()
        if current_row >= 0 and current_row < len(self.audio_files):
            self.current_index = current_row
            self.load_current_file()
            self.update_navigation_buttons()
    
    def show_previous(self):
        """前のファイルを表示"""
        if self.current_index > 0:
            self.file_list.setCurrentRow(self.current_index - 1)
    
    def show_next(self):
        """次のファイルを表示"""
        if self.current_index < len(self.audio_files) - 1:
            self.file_list.setCurrentRow(self.current_index + 1)
    
    def update_navigation_buttons(self):
        """ナビゲーションボタンの有効/無効を更新"""
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.audio_files) - 1)
    
    def load_current_file(self):
        """現在選択されているファイルを読み込む"""
        try:
            file_path = self.audio_files[self.current_index]
            self.audio_data, self.sr = sf.read(file_path)
            
            # ファイル情報の更新
            duration = len(self.audio_data) / self.sr
            self.info_label.setText(
                f'ファイル名: {os.path.basename(file_path)}\n'
                f'サンプリングレート: {self.sr} Hz\n'
                f'持続時間: {duration:.3f} 秒'
            )
            
            self.update_plot()
            
        except Exception as e:
            self.info_label.setText(f'エラー: {str(e)}')
    
    def update_stft_params(self):
        """STFTパラメータの更新"""
        self.n_fft = self.fft_size.value()
        self.hop_length = self.hop_size.value()
        self.win_length = self.win_size.value()
        self.update_plot()
    
    def update_freq_range(self):
        """周波数範囲の更新"""
        self.freq_max = self.freq_max_spin.value()
        self.update_plot()
    
    def update_plot(self):
        """波形とスペクトログラムの表示を更新"""
        if self.audio_data is None:
            return
        
        # グラフのクリア
        self.ax_wave.clear()
        self.ax_spec.clear()
        self.cax.clear()
        
        # 波形の表示
        time = np.arange(len(self.audio_data)) / self.sr
        self.ax_wave.plot(time, self.audio_data)
        self.ax_wave.set_title('波形')
        self.ax_wave.set_xlabel('時間 (秒)')
        self.ax_wave.set_ylabel('振幅')
        self.ax_wave.grid(True)
        
        # スペクトログラムの表示
        D = librosa.stft(self.audio_data,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # 周波数範囲の設定
        freq_ratio = self.freq_max / (self.sr/2)
        freq_bins = int(D_db.shape[0] * freq_ratio)
        
        img = self.ax_spec.imshow(D_db[:freq_bins],
                                aspect='auto',
                                origin='lower',
                                extent=[0, len(self.audio_data)/self.sr,
                                       self.freq_min,
                                       self.freq_max],
                                cmap='magma')
        
        self.ax_spec.set_title('スペクトログラム')
        self.ax_spec.set_xlabel('時間 (秒)')
        self.ax_spec.set_ylabel('周波数 (Hz)')
        
        # カラーバーの更新
        plt.colorbar(img, cax=self.cax, format='%+2.0f dB')
        
        # レイアウトの更新
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SegmentViewer()
    window.show()
    sys.exit(app.exec()) 