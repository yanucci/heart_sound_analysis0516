# 心音データ解析ツール

## クイックスタート

心音データ解析の基本的な流れは以下の3つのスクリプトを順番に実行します：

1. データセット登録
```bash
cd /Users/yusuke.yano/python_practice/heart_sound_analysis && python register_input.py
```
※ ファイル選択ダイアログが開くので、data/inputディレクトリから登録したいWAVファイルを選択してください。

2. ピーク検出
```bash
cd /Users/yusuke.yano/python_practice/heart_sound_analysis && python manual_peak_input_v5.py
```

3. セグメント分割・選別
```bash
cd /Users/yusuke.yano/python_practice/heart_sound_analysis && python visualize_segments_v3.py
```

## 概要
このプロジェクトは心音データの解析と可視化を行うためのツール群です。

## 主要機能

### 1. セグメント波形表示ツール (visualize_segments_v3.py)

心音データのセグメント分割と可視化を行うGUIツールです。

#### 機能
- 3段表示（全体波形、詳細波形、スペクトログラム）
- データセットの選択と自動読み込み
- セグメントの採用/不採用の管理
- セグメントの自動フィルタリング（持続時間による）
- セグメントの保存機能

#### 使用方法
1. ツールの起動
```bash
python visualize_segments_v3.py
```

2. データセットの選択
- コンボボックスから対象のデータセットを選択
- 音声ファイルとピークデータが自動的に読み込まれる

3. セグメントの確認と編集
- セグメントリストから対象を選択して波形を確認
- チェックボックスで採用/不採用を設定
- 持続時間が中央値の1/2未満または2倍超のセグメントは自動的に不採用

4. セグメントの保存
- 「セグメントを保存」ボタンで採用したセグメントを保存
- 保存先: `data/segments/{データセット名}_segment_*.wav`
- ログファイル: `data/segments/{データセット名}_segments_log.txt`

### 2. セグメント確認ツール (view_segments.py)

※ `view_segments.py` は現在リポジトリに存在しません（削除済み）。

---

## ディレクトリ構造
```
heart_sound_analysis/
├── data/
│   ├── input/                 # 入力音声ファイル
│   ├── peak_logs/            # ピークデータ
│   ├── segments/             # 保存されたセグメント
│   └── dataset_index.csv     # データセット情報
├── manual_peak_input_v5.py   # ピーク検出GUI
├── register_input.py         # 入力ファイル登録
├── visualize_segments_v3.py  # セグメント可視化・選別GUI
└── ...（src/や他の補助モジュール）
```

## 依存パッケージ
- Python 3.9+
- numpy
- pandas
- librosa
- soundfile
- matplotlib
- PyQt6

## インストール
```bash
pip install numpy pandas librosa soundfile matplotlib PyQt6
```

## 心音データ解析の手順

### 1. ピーク検出（manual_peak_input_v5.py）

心音データの第一心音と第二心音のピークを手動で入力するためのツールです。

#### 機能
- 3段表示（全体波形、詳細波形、スペクトログラム）
- マウスクリックによるピーク入力
- ピークの追加/削除
- ピークデータの保存

#### 使用方法
1. ツールの起動
```bash
python manual_peak_input_v5.py
```

2. 音声ファイルの読み込み
- 「Open WAV File」ボタンで対象の音声ファイルを選択
- data/inputディレクトリから選択

3. ピークの入力
- 波形またはスペクトログラム上で左クリックしてピークを追加
- 右クリックで最も近いピークを削除
- 左右矢印キーで表示位置を移動
- Cmd+Z/Ctrl+Zで直前の操作を取り消し

4. ピークの保存
- 「Save Peaks」ボタンでピークデータを保存
- 保存先: `data/peak_logs/{データセット名}_peaks.csv`

### 2. セグメント分割と可視化（visualize_segments_v3.py）

ピーク検出で得られたデータを基に、心音データをセグメントに分割し、可視化・選別を行うツールです。

#### 使用手順
1. ピーク検出ツール（manual_peak_input_v5.py）でピークを入力
2. セグメント可視化ツール（visualize_segments_v3.py）を起動
3. データセットを選択して波形とセグメントを表示
4. 各セグメントの採用/不採用を決定
5. 選別したセグメントを保存

この手順により、心音データから個々の心音セグメントを抽出し、品質の良いセグメントのみを選別することができます。 