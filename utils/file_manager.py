"""
ファイル管理ユーティリティ
心音データの入出力とファイル管理を担当
"""

import os
import pandas as pd
import soundfile as sf
from datetime import datetime
import shutil

class HeartSoundFileManager:
    def __init__(self, base_dir):
        """
        初期化
        Args:
            base_dir (str): プロジェクトのルートディレクトリ
        """
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.input_dir = os.path.join(self.data_dir, 'input')
        self.peak_logs_dir = os.path.join(self.data_dir, 'peak_logs')
        self.segments_dir = os.path.join(self.data_dir, 'segments')
        self.output_done_dir = os.path.join(self.data_dir, 'output_done')
        self.dataset_index_path = os.path.join(self.data_dir, 'dataset_index.csv')
        
        # 必要なディレクトリの存在確認と作成
        self._ensure_directories()
        
    def _ensure_directories(self):
        """必要なディレクトリの存在確認と作成"""
        for directory in [self.input_dir, self.peak_logs_dir, self.segments_dir, self.output_done_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def register_input_file(self, file_path, description=""):
        """
        入力ファイルの登録
        Args:
            file_path (str): 登録するWAVファイルのパス
            description (str): データの説明
        """
        # ファイル情報の取得
        name = os.path.splitext(os.path.basename(file_path))[0]
        audio_info = sf.info(file_path)
        
        # データセットインデックスの更新
        if os.path.exists(self.dataset_index_path):
            df = pd.read_csv(self.dataset_index_path)
        else:
            df = pd.DataFrame(columns=[
                'name', 'date', 'description', 'status',
                'sampling_rate', 'channels', 'format',
                'processing_date', 'completion_date'
            ])
        
        new_data = pd.DataFrame([{
            'name': name,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'description': description,
            'status': 'raw',
            'sampling_rate': audio_info.samplerate,
            'channels': audio_info.channels,
            'format': audio_info.format,
            'processing_date': None,
            'completion_date': None
        }])
        
        # 既存のデータを更新または新規追加
        if name in df['name'].values:
            df = df[df['name'] != name]  # 既存のエントリを削除
        df = pd.concat([df, new_data], ignore_index=True)
        
        # CSVファイルに保存
        df.to_csv(self.dataset_index_path, index=False)
        
        # ファイルを入力ディレクトリにコピー（同じファイルでない場合のみ）
        dest_path = os.path.join(self.input_dir, os.path.basename(file_path))
        if os.path.abspath(file_path) != os.path.abspath(dest_path):
            shutil.copy2(file_path, dest_path)
            print(f"ファイル {name} をコピーして登録しました。")
        else:
            print(f"ファイル {name} を登録しました（既に入力ディレクトリに存在）。")
        print(f"保存先: {dest_path}")
    
    def get_peak_log_path(self, name):
        """
        ピークログファイルのパスを取得
        Args:
            name (str): データセット名
        Returns:
            str: ピークログファイルのパス
        """
        return os.path.join(self.peak_logs_dir, f"{name}_peaks.csv")
    
    def get_segment_path(self, name, number):
        """
        セグメントファイルのパスを取得
        Args:
            name (str): データセット名
            number (int): セグメント番号
        Returns:
            str: セグメントファイルのパス
        """
        return os.path.join(self.segments_dir, f"{name}_segment_{number}.wav")
    
    def update_status(self, name, status):
        """
        データセットのステータスを更新
        Args:
            name (str): データセット名
            status (str): 新しいステータス
        """
        df = pd.read_csv(self.dataset_index_path)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if status == 'processing':
            df.loc[df['name'] == name, 'processing_date'] = current_time
        elif status == 'completed':
            df.loc[df['name'] == name, 'completion_date'] = current_time
            self.move_to_output_done(name)
            
        df.loc[df['name'] == name, 'status'] = status
        df.to_csv(self.dataset_index_path, index=False)
    
    def move_to_output_done(self, name):
        """
        処理完了したファイルをoutput_doneディレクトリに移動
        Args:
            name (str): データセット名
        """
        source_path = os.path.join(self.input_dir, f"{name}.wav")
        dest_path = os.path.join(self.output_done_dir, f"{name}.wav")
        
        if os.path.exists(source_path):
            shutil.move(source_path, dest_path)
    
    def get_dataset_info(self, name):
        """
        データセットの情報を取得
        Args:
            name (str): データセット名
        Returns:
            dict: データセットの情報
        """
        df = pd.read_csv(self.dataset_index_path)
        if name in df['name'].values:
            return df[df['name'] == name].to_dict('records')[0]
        return None
    
    def get_processing_status(self, name):
        """
        データセットの処理状態を取得
        Args:
            name (str): データセット名
        Returns:
            dict: 処理状態の情報
        """
        info = self.get_dataset_info(name)
        if info is None:
            return None
            
        return {
            'status': info['status'],
            'processing_date': info['processing_date'],
            'completion_date': info['completion_date']
        } 