import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils_preprocessing import bandpass_filter, normalize_audio, minmax_scale_audio, rescale_duration, extract_mfcc
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

data_dir = '/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments'
csv_path = '/Users/yusuke.yano/python_practice/heart_sound_analysis/data/dataset_index.csv'
wav_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]

# CSVからラベル一覧取得
df_index = pd.read_csv(csv_path)
name_set = set(df_index['name'].values)

features = []
labels = []
for wav_path in wav_files:
    fname = os.path.basename(wav_path)
    label = fname.split('_')[0]
    if label not in name_set:
        continue
    audio, sr = librosa.load(wav_path, sr=5000)
    audio = bandpass_filter(audio, sr)
    audio = normalize_audio(audio)
    audio = minmax_scale_audio(audio)
    audio = rescale_duration(audio, target_length=512)
    mfcc = extract_mfcc(audio, sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=0)
    features.append(mfcc_mean)
    labels.append(label)
features = np.array(features)
labels = np.array(labels)

# ラベルを数値に変換
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels_num = le.fit_transform(labels)

# 訓練・テスト分割
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(features, labels_num, test_size=test_size, random_state=0, stratify=labels_num)

# 分類器学習
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# 予測・評価
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc:.3f}')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification report:')
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 混同行列可視化
import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# 各クラスごとのMFCC平均プロット
plt.figure(figsize=(10,4))
for i, cname in enumerate(le.classes_):
    mean_mfcc = features[labels_num==i].mean(axis=0)
    plt.plot(mean_mfcc, label=cname)
plt.legend()
plt.title('Class-wise Mean MFCC')
plt.savefig('class_mean_mfcc.png')
plt.show()

# PCA 2次元プロット
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(features)
plt.figure(figsize=(8,6))
for i, cname in enumerate(le.classes_):
    plt.scatter(X_pca2[labels_num==i, 0], X_pca2[labels_num==i, 1], label=cname, alpha=0.7)
plt.legend()
plt.title('PCA of MFCC Features (2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('pca_2d.png')
plt.show()

# PCA 3次元プロット
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(features)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
for i, cname in enumerate(le.classes_):
    ax.scatter(X_pca3[labels_num==i, 0], X_pca3[labels_num==i, 1], X_pca3[labels_num==i, 2], label=cname, alpha=0.7)
ax.set_title('PCA of MFCC Features (3D)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend()
plt.savefig('pca_3d.png')
plt.show()

np.savez('pca3d_result.npz', X_pca3=X_pca3, labels=labels, labels_num=labels_num, class_names=le.classes_)
print('pca3d_result.npz を保存しました') 