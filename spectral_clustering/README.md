# 心音スペクトルクラスタリングプロジェクト

## 概要

このプロジェクトは、心音信号をスペクトル特徴量化し、教師なしクラスタリング（Spectral Clustering）で分類・可視化するものです。

- データ: `/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments` のwavファイル
- 主な流れ:
    1. 前処理（ノイズ除去・正規化）
    2. スペクトル特徴量抽出（STFT/MFCC）
    3. 類似度行列作成
    4. グラフラプラシアン・固有ベクトル抽出
    5. k-means等でクラスタリング
    6. 可視化

## ファイル構成

- `spectral_clustering_heart_sound.py` : メインスクリプト
- `utils_preprocessing.py` : 前処理・特徴量抽出用関数
- `requirements.txt` : 必要なPythonパッケージ

## 実行方法

```bash
python spectral_clustering_heart_sound.py
```

---

ご要望に応じて機能追加・修正可能です。 