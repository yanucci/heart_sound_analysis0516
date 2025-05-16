import sys
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QListWidget, QLabel,
                           QSpinBox, QDoubleSpinBox, QGroupBox, QComboBox,
                           QCheckBox, QFileDialog, QTableWidget, QTableWidgetItem,
                           QMessageBox)
from PyQt6.QtCore import Qt
import glob

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'

class SegmentVisualizerV3(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('セグメント波形表示 v3')
        self.setGeometry(100, 100, 1600, 900)
        
        # データ変数
        self.audio_data = None
        self.sr = None
        self.peaks = None
        self.segments = []
        self.segment_status = {}  # セグメントの採用/不採用状態
        self.dataset_info = None  # データセット情報
        
        # ベースディレクトリの設定
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(current_dir) == 'heart_sound_analysis':
            self.base_dir = current_dir
        else:
            self.base_dir = os.path.dirname(current_dir)
        
        # STFTパラメータ
        self.n_fft = 2048
        self.hop_length = 128
        self.win_length = 1024
        
        # 表示パラメータ
        self.current_segment = None
        self.freq_min = 0
        self.freq_max = 5000
        
        # UIの初期化
        self.init_ui()
        
        # データセット情報の読み込み
        self.load_dataset_info()
        
    def init_ui(self):
        # メインウィジェット
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 左側のコントロールパネル
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # ファイル操作グループ
        file_group = QGroupBox("ファイル操作")
        file_layout = QVBoxLayout()
        
        # データセット選択コンボボックス
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_selected)
        file_layout.addWidget(QLabel('データセット:'))
        file_layout.addWidget(self.dataset_combo)
        
        # 従来のボタンは残すが、非表示に
        self.load_audio_button = QPushButton('音声ファイルを開く')
        self.load_audio_button.clicked.connect(self.load_audio_manual)
        self.load_audio_button.hide()
        
        self.load_peaks_button = QPushButton('ピークデータを開く')
        self.load_peaks_button.clicked.connect(self.load_peaks_manual)
        self.load_peaks_button.hide()
        
        save_segments_button = QPushButton('セグメントを保存')
        save_segments_button.clicked.connect(self.save_segments)
        file_layout.addWidget(save_segments_button)
        
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # セグメントリスト
        segment_group = QGroupBox("セグメントリスト")
        segment_layout = QVBoxLayout()
        
        self.segment_list = QTableWidget()
        self.segment_list.setColumnCount(4)
        self.segment_list.setHorizontalHeaderLabels(['#', '時間範囲', '持続時間(秒)', '採用'])
        self.segment_list.itemSelectionChanged.connect(self.on_segment_selected)
        segment_layout.addWidget(self.segment_list)
        
        segment_group.setLayout(segment_layout)
        left_layout.addWidget(segment_group)
        
        # STFTパラメータ設定
        stft_group = QGroupBox("STFTパラメータ")
        stft_layout = QVBoxLayout()
        
        # FFTサイズ
        fft_layout = QHBoxLayout()
        fft_layout.addWidget(QLabel('FFTサイズ:'))
        self.fft_size = QSpinBox()
        self.fft_size.setRange(256, 4096)
        self.fft_size.setValue(self.n_fft)
        self.fft_size.valueChanged.connect(self.update_stft_params)
        fft_layout.addWidget(self.fft_size)
        stft_layout.addLayout(fft_layout)
        
        # ホップ長
        hop_layout = QHBoxLayout()
        hop_layout.addWidget(QLabel('ホップ長:'))
        self.hop_size = QSpinBox()
        self.hop_size.setRange(64, 1024)
        self.hop_size.setValue(self.hop_length)
        self.hop_size.valueChanged.connect(self.update_stft_params)
        hop_layout.addWidget(self.hop_size)
        stft_layout.addLayout(hop_layout)
        
        # 窓長
        win_layout = QHBoxLayout()
        win_layout.addWidget(QLabel('窓長:'))
        self.win_size = QSpinBox()
        self.win_size.setRange(256, 4096)
        self.win_size.setValue(self.win_length)
        self.win_size.valueChanged.connect(self.update_stft_params)
        win_layout.addWidget(self.win_size)
        stft_layout.addLayout(win_layout)
        
        stft_group.setLayout(stft_layout)
        left_layout.addWidget(stft_group)
        
        main_layout.addWidget(left_panel, stretch=1)
        
        # 右側のグラフ表示部分
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # グラフ表示用のmatplotlibキャンバス
        self.figure = plt.figure(figsize=(12, 10))
        gs = self.figure.add_gridspec(3, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1])
        
        # 全体波形、詳細波形、スペクトログラムのサブプロット
        self.ax_overview = self.figure.add_subplot(gs[0, 0])  # 全体波形
        self.ax_wave = self.figure.add_subplot(gs[1, 0])      # 詳細波形
        self.ax_spec = self.figure.add_subplot(gs[2, 0])      # スペクトログラム
        self.cax = self.figure.add_subplot(gs[2, 1])          # カラーバー
        
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, right_panel)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        main_layout.addWidget(right_panel, stretch=4)
    
    def load_dataset_info(self):
        print('base_dir:', self.base_dir)
        csv_path = os.path.join(self.base_dir, 'data', 'dataset_index.csv')
        print('csv_path:', csv_path)
        self.dataset_info = pd.read_csv(csv_path)
        
        # コンボボックスにデータセット名を追加
        self.dataset_combo.clear()
        self.dataset_combo.addItem('-- データセットを選択 --')  # 空の選択肢を追加
        self.dataset_combo.addItems(self.dataset_info['name'].tolist())
        
    def on_dataset_selected(self, index):
        """データセットが選択されたときの処理"""
        if index <= 0 or self.dataset_info is None:  # インデックス0（空の選択肢）の場合は処理しない
            # 表示をクリア
            self.audio_data = None
            self.sr = None
            self.peaks = None
            self.segments.clear()
            self.segment_status.clear()
            self.current_segment = None
            self.current_dataset_name = None
            self.update_segment_list()
            self.update_plot()
            return
            
        self.current_dataset_name = self.dataset_combo.currentText()
        
        # 音声ファイルのパス
        audio_path = os.path.join(self.base_dir, 'data', 'input', f"{self.current_dataset_name}.wav")
        # ピークファイルのパス
        peak_path = '/Users/yusuke.yano/python_practice/heart_sound_analysis/data/peak_logs/{}_peaks.csv'.format(self.current_dataset_name)
        
        try:
            # 音声ファイルの読み込み
            if os.path.exists(audio_path):
                self.audio_data, self.sr = librosa.load(audio_path, sr=None)
            else:
                raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")
            
            # ピークファイルの読み込み
            if os.path.exists(peak_path):
                self.peaks = pd.read_csv(peak_path)
                self.create_segments()
            else:
                raise FileNotFoundError(f"ピークファイルが見つかりません: {peak_path}")
                
        except Exception as e:
            QMessageBox.warning(self, 'エラー', f'ファイルの読み込みに失敗しました: {str(e)}')

    def create_segments(self):
        if self.audio_data is None or self.peaks is None:
            return
        
        # セグメントをクリア
        self.segments.clear()
        self.segment_status.clear()
        
        # ピークの時間でソート
        sorted_peaks = self.peaks.sort_values('time')
        peak_times = sorted_peaks['time'].values
        
        # セグメントの作成と持続時間の計算
        durations = []
        for i in range(len(peak_times) - 1):
            start_time = peak_times[i]
            end_time = peak_times[i + 1]
            duration = end_time - start_time
            durations.append(duration)
            
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            
            segment = {
                'start_time': start_time,
                'end_time': end_time,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'audio_data': self.audio_data[start_sample:end_sample],
                'number': i + 1,
                'duration': duration
            }
            self.segments.append(segment)
        
        # 持続時間の中央値を計算
        median_duration = np.median(durations)
        
        # セグメントの採用/不採用を設定
        for i, segment in enumerate(self.segments):
            # 持続時間が中央値の1/2未満または2倍超の場合は不採用
            is_valid = 0.5 * median_duration <= segment['duration'] <= 2.0 * median_duration
            self.segment_status[i] = is_valid
        
        self.update_segment_list()
        if self.segments:
            self.segment_list.selectRow(0)
    
    def update_segment_list(self):
        self.segment_list.setRowCount(len(self.segments))
        for i, segment in enumerate(self.segments):
            # セグメント番号
            self.segment_list.setItem(i, 0, QTableWidgetItem(str(segment['number'])))
            
            # 時間範囲
            time_range = f"{segment['start_time']:.2f}s - {segment['end_time']:.2f}s"
            self.segment_list.setItem(i, 1, QTableWidgetItem(time_range))
            
            # 持続時間
            duration = segment['end_time'] - segment['start_time']
            self.segment_list.setItem(i, 2, QTableWidgetItem(f"{duration:.2f}"))
            
            # 採用/不採用チェックボックス
            checkbox = QCheckBox()
            checkbox.setChecked(bool(self.segment_status[i]))  # bool型に明示的に変換
            checkbox.stateChanged.connect(lambda state, row=i: self.on_segment_status_changed(row, state))
            self.segment_list.setCellWidget(i, 3, checkbox)
        
        # 列幅の調整
        self.segment_list.resizeColumnsToContents()
    
    def on_segment_selected(self):
        selected_items = self.segment_list.selectedItems()
        if not selected_items:
            return
        
        row = selected_items[0].row()
        self.current_segment = self.segments[row]
        self.update_plot()
    
    def on_segment_status_changed(self, row, state):
        self.segment_status[row] = bool(state)
    
    def update_stft_params(self):
        self.n_fft = self.fft_size.value()
        self.hop_length = self.hop_size.value()
        self.win_length = self.win_size.value()
        self.update_plot()
    
    def update_plot(self):
        if self.current_segment is None:
            return
        
        # グラフのクリア
        self.ax_overview.clear()
        self.ax_wave.clear()
        self.ax_spec.clear()
        self.cax.clear()
        
        # 全体波形の表示
        time = np.arange(len(self.audio_data)) / self.sr
        self.ax_overview.plot(time, self.audio_data, 'b-', linewidth=0.5, alpha=0.7)
        self.ax_overview.set_title('全体波形')
        self.ax_overview.set_xlabel('時間 (秒)')
        self.ax_overview.set_ylabel('振幅')
        self.ax_overview.grid(True)
        
        # 全体波形に全セグメントの境界とステータスを表示
        for segment in self.segments:
            # セグメントの開始位置
            self.ax_overview.axvline(x=segment['start_time'], 
                                   color='g', linestyle='--', alpha=0.5)
            
            # セグメント番号を表示
            self.ax_overview.text(segment['start_time'], 
                                self.ax_overview.get_ylim()[1],
                                f'S{segment["number"]}',
                                horizontalalignment='right',
                                verticalalignment='bottom')
            
            # 不採用セグメントに×を表示
            if not self.segment_status[segment['number']-1]:
                center_time = (segment['start_time'] + segment['end_time']) / 2
                y_pos = self.ax_overview.get_ylim()[1] * 0.8
                self.ax_overview.plot(center_time, y_pos, 'rx', markersize=10)
                self.ax_overview.text(center_time, y_pos, '不採用',
                                    color='red',
                                    horizontalalignment='center',
                                    verticalalignment='bottom')
        
        # 現在のセグメントをハイライト
        self.ax_overview.axvspan(self.current_segment['start_time'],
                               self.current_segment['end_time'],
                               color='r', alpha=0.2)
        
        # 詳細波形の表示（現在のセグメント）
        segment_time = np.arange(len(self.current_segment['audio_data'])) / self.sr
        segment_time += self.current_segment['start_time']  # 絶対時間に変換
        self.ax_wave.plot(segment_time, self.current_segment['audio_data'])
        duration = self.current_segment['end_time'] - self.current_segment['start_time']
        self.ax_wave.set_title(f'セグメント {self.current_segment["number"]} の波形 ' + 
                              f'({self.current_segment["start_time"]:.2f}s - {self.current_segment["end_time"]:.2f}s, ' +
                              f'持続時間: {duration:.2f}s)')
        self.ax_wave.set_xlabel('時間 (秒)')
        self.ax_wave.set_ylabel('振幅')
        self.ax_wave.grid(True)
        
        # スペクトログラムの表示
        D = librosa.stft(self.current_segment['audio_data'],
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
                                extent=[self.current_segment['start_time'],
                                       self.current_segment['end_time'],
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
    
    def save_segments(self):
        """採用されたセグメントを保存し、ログを出力する"""
        if not self.segments:
            QMessageBox.warning(self, '警告', 'セグメントが存在しません。')
            return
        
        if not hasattr(self, 'current_dataset_name'):
            QMessageBox.warning(self, '警告', 'データセットが選択されていません。')
            return

        # 保存先ディレクトリの設定
        save_dir = os.path.join(self.base_dir, 'data', 'segments')
        os.makedirs(save_dir, exist_ok=True)

        try:
            # 既存のセグメントファイルを削除
            existing_pattern = os.path.join(save_dir, f"{self.current_dataset_name}_segment_*.wav")
            for file in glob.glob(existing_pattern):
                os.remove(file)

            # ログファイルの準備
            log_path = os.path.join(save_dir, f"{self.current_dataset_name}_segments_log.txt")
            saved_count = 0
            
            with open(log_path, 'w', encoding='utf-8') as log_file:
                log_file.write(f"セグメント保存ログ - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"データセット: {self.current_dataset_name}\n")
                log_file.write("-" * 50 + "\n")
                
                # 採用されたセグメントのみを保存
                for i, segment in enumerate(self.segments):
                    if self.segment_status[i]:
                        filename = f"{self.current_dataset_name}_segment_{segment['number']}.wav"
                        save_path = os.path.join(save_dir, filename)
                        
                        # WAVファイルとして保存
                        sf.write(save_path, segment['audio_data'], self.sr)
                        
                        # ログに記録
                        log_file.write(f"保存完了: {filename}\n")
                        log_file.write(f"  開始時間: {segment['start_time']:.3f}秒\n")
                        log_file.write(f"  終了時間: {segment['end_time']:.3f}秒\n")
                        log_file.write(f"  持続時間: {segment['duration']:.3f}秒\n")
                        log_file.write("-" * 30 + "\n")
                        saved_count += 1
                
                # 保存の概要を記録
                log_file.write(f"\n保存概要:\n")
                log_file.write(f"合計セグメント数: {len(self.segments)}\n")
                log_file.write(f"保存したセグメント数: {saved_count}\n")
            
            QMessageBox.information(self, '完了', 
                                  f'セグメントの保存が完了しました。\n'
                                  f'保存先: {save_dir}\n'
                                  f'保存数: {saved_count}個\n'
                                  f'詳細はログファイルを確認してください。')
            
        except Exception as e:
            QMessageBox.critical(self, 'エラー', 
                               f'セグメントの保存中にエラーが発生しました。\n{str(e)}')

    def load_audio_manual(self):
        """手動での音声ファイル選択（バックアップ用）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '音声ファイルを選択',
            os.path.join(self.base_dir, 'data', 'input'),
            'WAV files (*.wav)'
        )
        
        if file_path:
            self.audio_data, self.sr = librosa.load(file_path, sr=None)
            if self.peaks is not None:
                self.create_segments()

    def load_peaks_manual(self):
        """手動でのピークファイル選択（バックアップ用）"""
        initial_dir = '/Users/yusuke.yano/python_practice/heart_sound_analysis/data/peak_logs'
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'ピークデータを選択',
            initial_dir,
            'CSV files (*.csv)'
        )
        
        if file_path:
            self.peaks = pd.read_csv(file_path)
            if self.audio_data is not None:
                self.create_segments()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SegmentVisualizerV3()
    window.show()
    sys.exit(app.exec()) 