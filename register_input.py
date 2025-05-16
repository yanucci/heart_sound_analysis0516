"""
入力ファイル登録スクリプト
GUIでWAVファイルを選択して登録します。
"""

from utils.file_manager import HeartSoundFileManager
import os
import sys
from PyQt6.QtWidgets import QApplication, QFileDialog

def main():
    # ベースディレクトリの設定
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "data", "input")
    
    # GUIアプリケーションの初期化
    app = QApplication(sys.argv)
    
    # ファイル選択ダイアログを表示
    file_dialog = QFileDialog()
    file_dialog.setDirectory(input_dir)  # 初期ディレクトリを設定
    file_dialog.setNameFilter("WAV files (*.wav)")  # WAVファイルのみ表示
    file_dialog.setWindowTitle("登録するWAVファイルを選択")
    
    if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
        # 選択されたファイルのパスを取得
        input_wav_path = file_dialog.selectedFiles()[0]
        
        # 説明文を生成（ファイル名を使用）
        description = f"{os.path.splitext(os.path.basename(input_wav_path))[0]} heart sound data"
        
        # ファイルマネージャーの初期化
        file_manager = HeartSoundFileManager(base_dir)
        
        try:
            # ファイルの登録
            file_manager.register_input_file(
                input_wav_path,
                description=description
            )
        except Exception as e:
            print(f"エラー: ファイルの登録に失敗しました。\n{str(e)}")
            sys.exit(1)
    else:
        print("ファイル選択がキャンセルされました。")
        sys.exit(0)

if __name__ == "__main__":
    main() 