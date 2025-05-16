# 心音データ分類・Grad-CAM可視化 最終まとめ

## 1. 概要
- 心音WAVデータを被験者ごとに分類するCNNモデルを構築
- スペクトログラム（STFT, dBスケール, 0-5000Hz）を入力特徴量として使用
- Grad-CAMによる可視化（ヒートマップ単体・重ね画像）に対応

## 2. ファイル構成
- `src/preprocess_and_classify.py` : メインスクリプト
- `data/segments/` : WAVデータ格納ディレクトリ
- `data/segments/gradcam_heatmap_only/` : Grad-CAMヒートマップ単体画像の出力先
- `data/segments/gradcam_masked/` : Grad-CAM重ね画像（マスク方式）の出力先

## 3. 実行方法
```bash
cd heart_sound_analysis
python src/preprocess_and_classify.py
```

## 4. 主な処理内容
- WAVファイルの前処理（リサンプリング, 正規化, 1秒パディング）
- STFTスペクトログラム（dBスケール, 0-5000Hz, 0-1正規化）生成
- CNNモデル（Functional API, Conv2D×3, Dense, Dropout）で分類
- モデル学習・保存・学習曲線出力
- Grad-CAMヒートマップ生成（全ファイル一括）
- ヒートマップ単体画像/重ね画像（マスク方式）をフォルダに保存

## 5. 出力例
- `training_history.png` : 学習曲線
- `gradcam_heatmap_only/gradcam_heatmap_xxx.png` : ヒートマップ単体
- `gradcam_masked/gradcam_masked_xxx.png` : スペクトログラム＋Grad-CAM重ね画像

## 6. 注意点・Tips
- ヒートマップ単体はしっかり可視化できるが、重ね画像はalphaやマスクしきい値の調整が必要
- モデルの学習が不十分な場合、ヒートマップが全体的に薄くなることがある
- `librosa`の警告（Empty filters detected）は無視してOK
- 実験パラメータ（しきい値、カラーマップ、alpha等）は`preprocess_and_classify.py`内で調整可能

## 7. Grad-CAMヒートマップが全て0.0になる現象について

### 実行例（出力ログ抜粋）
```
ashiharahLSB_segment_14.wav: heatmap min=0.1963, max=0.9650, mean=0.2195, std=0.0701
[GradCAM] predictions: [[0.12170139 0.09421796 0.24103336 ...]], class_idx: 2
[GradCAM] heatmap before ReLU: min=-0.0030, max=-0.0005, mean=-0.0006
[GradCAM] heatmap after ReLU: min=0.0, max=0.0, mean=0.0
[GradCAM] heatmap after norm: min=0.0, max=0.0, mean=0.0
...
```

### 解説・考察
- Grad-CAMの計算では、勾配と特徴マップの重み付き和を取った結果（heatmap）は正にも負にもなり得ます。
- 可視化では「どこがどれだけ貢献したか」を見たいので、負の値はReLU（0未満を0にする）でカットします。
- ReLU前のheatmapが全て負の場合、ReLU後は全て0.0となり、正規化しても全体が0.0のままになります。
- これは「その入力に対して、モデルが活性化（正の寄与）を全く示していない」ことを意味します。
- 学習の進行やパラメータの変化で、こうした現象が連続して起こることもあります。
- 0.0が多すぎる場合は、モデルの学習状況やデータ分布、またはGrad-CAMの計算層（layer_name）を見直すのも有効です。

---

何か追加でまとめたい内容や、他の出力例が必要な場合はご指示ください。 