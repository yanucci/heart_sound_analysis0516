"""
Heart Sound Peak Detection Tool
Interactive tool for detecting and annotating peaks in heart sound recordings.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QLineEdit,
                           QTableWidget, QTableWidgetItem, QFileDialog,
                           QSpinBox, QGroupBox, QSlider)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence
import librosa

from src.utils.file_manager import HeartSoundFileManager
from src.utils.data_processor import HeartSoundProcessor

# Set default font
plt.rcParams['font.family'] = 'Arial'

class PeakInputWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Heart Sound Peak Detection Tool')
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize file manager and processor
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.file_manager = HeartSoundFileManager(self.base_dir)
        self.processor = HeartSoundProcessor(self.file_manager)
        
        # Data variables
        self.file_name = None
        self.audio_data = None
        self.sr = None
        self.time = None
        self.peaks = []
        self.peaks_history = []  # アンドゥ用の履歴
        
        # STFT parameters
        self.n_fft = 2048
        self.hop_length = 128
        self.win_length = 1024
        
        # View range for detail plots
        self.view_start = 0
        self.view_duration = 5  # seconds
        self.zoom_factor = 1.0
        self.move_step = 0.5  # キーボードでの移動ステップ（秒）
        
        # Cursor lines
        self.wave_cursor = None
        self.spec_cursor = None
        
        # Mouse drag variables
        self.dragging = False
        self.drag_start_x = None
        self.drag_start_view = None
        self.drag_threshold = 0.01  # ドラッグ判定の閾値（秒）
        self.click_start_pos = None
        
        self.init_ui()
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        # グローバルショートカットとして設定
        self.left_shortcut = QShortcut(QKeySequence("Left"), self)
        self.left_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.left_shortcut.activated.connect(self.move_view_left)
        
        self.right_shortcut = QShortcut(QKeySequence("Right"), self)
        self.right_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.right_shortcut.activated.connect(self.move_view_right)
        
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.undo_shortcut.activated.connect(self.undo_last_peak)
        
    def move_view_left(self):
        if self.audio_data is not None:
            self.view_start = max(0, self.view_start - self.move_step)
            self.update_plot()
            
    def move_view_right(self):
        if self.audio_data is not None:
            total_duration = len(self.audio_data) / self.sr
            self.view_start = min(total_duration - self.view_duration, self.view_start + self.move_step)
            self.update_plot()

    def undo_last_peak(self):
        if self.peaks_history:
            self.peaks = self.peaks_history.pop()
            self.update_plot()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left control panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # File selection button
        file_button = QPushButton('Open WAV File')
        file_button.clicked.connect(self.load_file)
        file_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        left_layout.addWidget(file_button)
        
        # Instructions
        instructions_group = QGroupBox("How to Use")
        instructions_layout = QVBoxLayout()
        instructions = [
            "Left Click: Add peak marker",
            "Pan Tool (Hand icon): Click and drag to move view",
            "Right Click: Remove nearest peak",
            "Left/Right Arrow: Move view position",
            "Cmd+Z / Ctrl+Z: Undo last action",
            "Mouse Move: Show time and frequency",
            "Zoom Slider: Adjust detail view",
            "Save Peaks: Store peak data"
        ]
        for instruction in instructions:
            label = QLabel(instruction)
            instructions_layout.addWidget(label)
        instructions_group.setLayout(instructions_layout)
        left_layout.addWidget(instructions_group)
        
        # STFT parameters group
        stft_group = QGroupBox("STFT Parameters")
        stft_layout = QVBoxLayout()
        
        # FFT size
        fft_layout = QHBoxLayout()
        fft_layout.addWidget(QLabel('FFT Size:'))
        self.fft_size = QSpinBox()
        self.fft_size.setRange(256, 4096)
        self.fft_size.setValue(self.n_fft)
        self.fft_size.valueChanged.connect(self.update_stft_params)
        self.fft_size.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        fft_layout.addWidget(self.fft_size)
        stft_layout.addLayout(fft_layout)
        
        # Hop length
        hop_layout = QHBoxLayout()
        hop_layout.addWidget(QLabel('Hop Length:'))
        self.hop_size = QSpinBox()
        self.hop_size.setRange(64, 1024)
        self.hop_size.setValue(self.hop_length)
        self.hop_size.valueChanged.connect(self.update_stft_params)
        self.hop_size.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        hop_layout.addWidget(self.hop_size)
        stft_layout.addLayout(hop_layout)
        
        # Window length
        win_layout = QHBoxLayout()
        win_layout.addWidget(QLabel('Window Length:'))
        self.win_size = QSpinBox()
        self.win_size.setRange(256, 4096)
        self.win_size.setValue(self.win_length)
        self.win_size.valueChanged.connect(self.update_stft_params)
        self.win_size.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        win_layout.addWidget(self.win_size)
        stft_layout.addLayout(win_layout)
        
        stft_group.setLayout(stft_layout)
        left_layout.addWidget(stft_group)
        
        # Zoom control
        zoom_group = QGroupBox("Detail View Control")
        zoom_layout = QVBoxLayout()
        
        # Zoom slider
        zoom_slider_layout = QHBoxLayout()
        zoom_slider_layout.addWidget(QLabel('Zoom:'))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        self.zoom_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        zoom_slider_layout.addWidget(self.zoom_slider)
        zoom_layout.addLayout(zoom_slider_layout)
        
        zoom_group.setLayout(zoom_layout)
        left_layout.addWidget(zoom_group)
        
        # Time input
        time_input = QWidget()
        time_layout = QHBoxLayout(time_input)
        time_layout.addWidget(QLabel('Time (s):'))
        self.time_edit = QLineEdit()
        self.time_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        time_layout.addWidget(self.time_edit)
        left_layout.addWidget(time_input)
        
        # Cursor position display
        self.cursor_pos_label = QLabel('Cursor Position: -- s')
        left_layout.addWidget(self.cursor_pos_label)
        
        # Add/Delete buttons
        button_layout = QHBoxLayout()
        add_button = QPushButton('Add Peak')
        add_button.clicked.connect(self.add_peak)
        add_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button_layout.addWidget(add_button)
        
        delete_button = QPushButton('Delete Selected Peak')
        delete_button.clicked.connect(self.delete_selected_peak)
        delete_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button_layout.addWidget(delete_button)
        left_layout.addLayout(button_layout)
        
        # Save button
        save_button = QPushButton('Save Peaks')
        save_button.clicked.connect(self.save_peaks)
        save_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        left_layout.addWidget(save_button)
        
        # Peak list
        self.peak_table = QTableWidget()
        self.peak_table.setColumnCount(2)
        self.peak_table.setHorizontalHeaderLabels(['Peak #', 'Time (s)'])
        self.peak_table.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        left_layout.addWidget(self.peak_table)
        
        main_layout.addWidget(left_panel, stretch=1)
        
        # Right graph panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Matplotlib canvas
        self.figure = plt.figure(figsize=(12, 10))
        gs = self.figure.add_gridspec(3, 2, width_ratios=[1, 0.05], height_ratios=[1, 1.5, 1.5])
        
        # Overview plot
        self.ax_overview = self.figure.add_subplot(gs[0, 0])
        self.ax_overview.set_title('Full Waveform Overview')
        
        # Detail waveform plot
        self.ax_wave = self.figure.add_subplot(gs[1, 0])
        self.ax_wave.set_title('Detailed Waveform View')
        
        # Spectrogram plot
        self.ax_spec = self.figure.add_subplot(gs[2, 0])
        self.ax_spec.set_title('Spectrogram (0-5000 Hz)')
        
        # Colorbar
        self.cax = self.figure.add_subplot(gs[2, 1])
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toolbar = NavigationToolbar(self.canvas, right_panel)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        main_layout.addWidget(right_panel, stretch=4)
        
        # Connect mouse events
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        # Set initial focus to canvas
        self.canvas.setFocus()
        
    def on_mouse_move(self, event):
        if event.inaxes in [self.ax_wave, self.ax_spec] and self.audio_data is not None:
            # Update cursor position label
            self.cursor_pos_label.setText(f'Cursor Position: {event.xdata:.3f} s')
            
            # Handle dragging for view position
            if self.dragging and hasattr(event, 'button') and event.button == 1:  # 左クリックでドラッグ
                if self.drag_start_x is not None:
                    # ドラッグ距離が閾値を超えた場合のみ移動
                    if abs(event.xdata - self.click_start_pos) > self.drag_threshold:
                        dx = self.drag_start_x - event.xdata
                        new_start = self.drag_start_view + dx
                        total_duration = len(self.audio_data) / self.sr
                        self.view_start = max(0, min(new_start, total_duration - self.view_duration))
                        self.update_plot()
                    return
            
            # Remove old cursor lines
            if self.wave_cursor:
                self.wave_cursor.remove()
            if self.spec_cursor:
                self.spec_cursor.remove()
            
            # Draw new cursor lines
            if event.inaxes == self.ax_wave:
                self.wave_cursor = self.ax_wave.axvline(x=event.xdata, color='g', linestyle='-', alpha=0.5)
                self.spec_cursor = self.ax_spec.axvline(x=event.xdata, color='g', linestyle='-', alpha=0.5)
            elif event.inaxes == self.ax_spec:
                self.wave_cursor = self.ax_wave.axvline(x=event.xdata, color='g', linestyle='-', alpha=0.5)
                self.spec_cursor = self.ax_spec.axvline(x=event.xdata, color='g', linestyle='-', alpha=0.5)
            
            # Draw horizontal line in spectrogram for frequency
            if event.inaxes == self.ax_spec:
                freq = event.ydata
                if 0 <= freq <= 5000:
                    self.cursor_pos_label.setText(
                        f'Cursor Position: {event.xdata:.3f} s, Frequency: {freq:.1f} Hz'
                    )
            
            self.canvas.draw()
            
    def on_mouse_click(self, event):
        if event.inaxes in [self.ax_wave, self.ax_spec, self.ax_overview]:  # ax_overviewを追加
            if event.button == 1:  # 左クリック: ピークの追加
                if event.inaxes in [self.ax_wave, self.ax_spec]:  # ピーク追加は詳細ビューとスペクトログラムのみ
                    time = event.xdata
                    if 0 <= time <= self.time[-1]:
                        self.peaks_history.append(self.peaks.copy())
                        self.peaks.append(time)
                        self.peaks.sort()
                        self.time_edit.setText(f"{time:.3f}")
                        self.update_plot()
            elif event.button == 3:  # 右クリック: 最も近いピークの削除
                if self.peaks:
                    time = event.xdata
                    nearest_peak = min(self.peaks, key=lambda x: abs(x - time))
                    if abs(nearest_peak - time) < 0.3:
                        self.peaks_history.append(self.peaks.copy())
                        self.peaks.remove(nearest_peak)
                        self.update_plot()

    def on_mouse_release(self, event):
        if event.button == 1 and self.click_start_pos is not None:
            # ドラッグ距離が閾値未満の場合はピークを追加
            if abs(event.xdata - self.click_start_pos) <= self.drag_threshold:
                time = event.xdata
                if 0 <= time <= self.time[-1]:
                    self.peaks_history.append(self.peaks.copy())
                    self.peaks.append(time)
                    self.peaks.sort()
                    self.time_edit.setText(f"{time:.3f}")
                    self.update_plot()
        
        self.dragging = False
        self.drag_start_x = None
        self.drag_start_view = None
        self.click_start_pos = None

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Select WAV File',
            '/Users/yusuke.yano/python_practice/heart_sound_analysis/data/input',
            'All Files (*);;WAV files (*.wav)'
        )
        
        if file_path:
            self.file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.audio_data, self.sr = librosa.load(file_path, sr=None)
            self.time = np.arange(len(self.audio_data)) / self.sr
            self.peaks = []
            self.load_existing_peaks()
            self.update_plot()
            
    def update_stft_params(self):
        self.n_fft = self.fft_size.value()
        self.hop_length = self.hop_size.value()
        self.win_length = self.win_size.value()
        if self.audio_data is not None:
            self.update_plot()
            
    def load_existing_peaks(self):
        if self.file_name:
            peak_file = self.file_manager.get_peak_log_path(self.file_name)
            if os.path.exists(peak_file):
                df = pd.read_csv(peak_file)
                self.peaks = df['time'].tolist()
            
    def update_zoom(self):
        if self.audio_data is not None:
            self.zoom_factor = self.zoom_slider.value() / 100
            self.update_plot()

    def plot_peak_markers(self, ax, peak_time, peak_amp=None, is_overview=False):
        """ピークマーカーを描画する共通関数"""
        # 垂直線を描画
        ax.axvline(x=peak_time, color='r', linestyle='--', alpha=0.5)
        
        # オーバービューの場合は振幅を計算
        if is_overview:
            peak_amp = self.audio_data[int(peak_time * self.sr)]
        
        # 振幅が指定されている場合はマーカーとラベルを追加
        if peak_amp is not None:
            ax.plot(peak_time, peak_amp, 'ro', markersize=8)
            peak_idx = self.peaks.index(peak_time) + 1
            ax.annotate(f'P{peak_idx}', 
                       (peak_time, peak_amp),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center')

    def update_plot(self):
        if self.audio_data is None:
            return
            
        # Clear all axes
        self.ax_overview.clear()
        self.ax_wave.clear()
        self.ax_spec.clear()
        
        # Plot overview
        self.ax_overview.plot(self.time, self.audio_data, 'b-', linewidth=0.5)
        self.ax_overview.set_title('Full Waveform Overview')
        self.ax_overview.set_xlabel('Time (s)')
        self.ax_overview.set_ylabel('Amplitude')
        
        # Plot peaks in overview
        for peak_time in self.peaks:
            self.plot_peak_markers(self.ax_overview, peak_time, is_overview=True)
        
        # Calculate view range for detailed plots
        total_duration = len(self.audio_data) / self.sr
        self.view_duration = 5 / self.zoom_factor
        self.view_start = max(0, min(self.view_start, total_duration - self.view_duration))
        
        # Calculate indices for the view window
        view_start_idx = int(self.view_start * self.sr)
        view_end_idx = int((self.view_start + self.view_duration) * self.sr)
        
        # Ensure the view window is exactly aligned with the time axis
        view_start_time = view_start_idx / self.sr
        view_end_time = view_end_idx / self.sr
        
        # Plot detailed waveform with exact time range
        self.ax_wave.plot(self.time[view_start_idx:view_end_idx],
                         self.audio_data[view_start_idx:view_end_idx],
                         'b-', linewidth=1)
        self.ax_wave.set_title('Detailed Waveform View')
        self.ax_wave.set_xlabel('Time (s)')
        self.ax_wave.set_ylabel('Amplitude')
        self.ax_wave.set_xlim(view_start_time, view_end_time)
        
        # Plot peaks in detailed view
        for peak_time in self.peaks:
            if view_start_time <= peak_time <= view_end_time:
                peak_amp = self.audio_data[int(peak_time * self.sr)]
                self.plot_peak_markers(self.ax_wave, peak_time, peak_amp)
        
        # Calculate and plot spectrogram
        D = librosa.stft(self.audio_data[view_start_idx:view_end_idx],
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Plot spectrogram with fixed frequency range (0-5000 Hz)
        max_freq = 5000
        freq_ratio = max_freq / (self.sr/2)
        freq_bins = int(S_db.shape[0] * freq_ratio)
        
        img = self.ax_spec.imshow(S_db[:freq_bins],
                                aspect='auto',
                                origin='lower',
                                extent=[view_start_time,
                                       view_end_time,
                                       0,
                                       max_freq],
                                cmap='magma')
        
        # Plot peaks in spectrogram
        for peak_time in self.peaks:
            if view_start_time <= peak_time <= view_end_time:
                self.plot_peak_markers(self.ax_spec, peak_time)
        
        self.ax_spec.set_title('Spectrogram (0-5000 Hz)')
        self.ax_spec.set_xlabel('Time (s)')
        self.ax_spec.set_ylabel('Frequency (Hz)')
        
        # Update colorbar
        plt.colorbar(img, cax=self.cax, label='dB')
        
        # Show overview range
        self.ax_overview.axvspan(view_start_time,
                               view_end_time,
                               color='r', alpha=0.2)
        
        # Update peak table
        self.peak_table.setRowCount(len(self.peaks))
        for i, peak in enumerate(self.peaks):
            self.peak_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.peak_table.setItem(i, 1, QTableWidgetItem(f"{peak:.3f}"))
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def add_peak(self):
        try:
            time = float(self.time_edit.text())
            if 0 <= time <= self.time[-1]:
                self.peaks.append(time)
                self.peaks.sort()
                self.update_plot()
                self.time_edit.clear()
        except ValueError:
            print("Please enter a valid time value")
            
    def delete_selected_peak(self):
        selected_items = self.peak_table.selectedItems()
        if not selected_items:
            return
            
        row = selected_items[0].row()
        if 0 <= row < len(self.peaks):
            self.peaks.pop(row)
            self.update_plot()
            
    def save_peaks(self):
        if not self.peaks or self.file_name is None:
            print("No peaks to save")
            return
            
        peak_data = {
            'peak_number': list(range(1, len(self.peaks) + 1)),
            'time': self.peaks,
            'amplitude': [float(self.audio_data[int(p * self.sr)]) for p in self.peaks],
            'status': ['valid'] * len(self.peaks)
        }
        
        self.processor.save_peaks(self.file_name, peak_data)
        print("Peak data saved successfully")

    def keyPressEvent(self, event):
        # スーパークラスのキーイベントを呼び出し
        super().keyPressEvent(event)
        
        # 左右キーの処理
        if event.key() == Qt.Key.Key_Left:
            self.move_view_left()
        elif event.key() == Qt.Key.Key_Right:
            self.move_view_right()

def main():
    app = QApplication(sys.argv)
    window = PeakInputWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 