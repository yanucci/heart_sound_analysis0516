# 心音教師あり分類プロジェクト

## 概要

心音信号のファイル名先頭（例：ashiharahLSB）をクラスラベルとし、音響特徴量（MFCC）から多クラス分類を行います。

- データ: `/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments` のwavファイル
- ラベル: `dataset_index.csv` のname列
- 主な流れ:
    1. 前処理・特徴量抽出
    2. ラベル付与（ファイル名先頭とcsv突合）
    3. 学習・評価
    4. 可視化

## ファイル構成
- `supervised_classification_heart_sound.py` : メインスクリプト
- `utils_preprocessing.py` : 前処理・特徴量抽出（流用）
- `requirements.txt` : 必要なPythonパッケージ

## 実行方法
```bash
python supervised_classification_heart_sound.py
```

## 追加機能・可視化・分析の流れ（2025-05-15追記）

### 1. PCAによる特徴量可視化
- MFCC特徴量をPCAで2次元・3次元に圧縮し、クラスごとに色分けして可視化
- 2D: `pca_2d.png`、3D: `pca_3d.png`として保存
- 3DプロットはmacOSのAquaウィンドウやJupyter Notebook（%matplotlib notebook）で回転可能

### 2. 3D PCA結果の保存と再利用
- 3次元PCA座標・ラベル・クラス名を`pca3d_result.npz`として保存
- これにより、あとから何度でも3D分布を再描画・分析可能

### 3. Plotlyによるインタラクティブ3D可視化
- `pca3d_plotly.py`で`pca3d_result.npz`を読み込み、Plotlyで3Dインタラクティブ表示
- Webブラウザ上で自由に回転・ズーム・クラスごとに色分け表示が可能
- コマンド例：
  ```bash
  python pca3d_plotly.py
  ```

### 4. 注意点
- Jupyter Notebookで3D回転がうまく動かない場合は、Plotlyスクリプト推奨
- `features.npy`や`labels.npy`も保存しておくと、特徴量やラベルの再利用が容易

---

ご要望に応じてさらなる可視化・分析機能も追加可能です。 