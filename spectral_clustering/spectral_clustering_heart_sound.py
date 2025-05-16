import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import silhouette_score
from scipy.linalg import eigh
from utils_preprocessing import bandpass_filter, normalize_audio, minmax_scale_audio, rescale_duration, extract_mfcc

data_dir = '/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments'
wav_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]

features = []
for wav_path in wav_files:
    audio, sr = librosa.load(wav_path, sr=5000)
    audio = bandpass_filter(audio, sr)
    audio = normalize_audio(audio)
    audio = minmax_scale_audio(audio)
    audio = rescale_duration(audio, target_length=512)
    mfcc = extract_mfcc(audio, sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=0)
    features.append(mfcc_mean)
features = np.array(features)

# --- クラスタ数kの最適化 ---
max_k = 8
sse = []
silhouette = []
k_range = range(2, max_k+1)
for k in k_range:
    affinity = pairwise_kernels(features, metric='rbf', gamma=0.5)
    D = np.diag(affinity.sum(axis=1))
    L = D - affinity
    vals, vecs = eigh(L, D)
    X_spec = vecs[:, 1:k]  # 0番目は定数ベクトルなので除外
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_spec)
    sse.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_spec, kmeans.labels_))

plt.figure()
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.savefig('elbow_method.png')
plt.close()

plt.figure()
plt.plot(k_range, silhouette, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.savefig('silhouette_score.png')
plt.close()

# 最適kを自動選択（シルエットスコア最大のk）
k_opt = k_range[np.argmax(silhouette)]
print(f'推奨クラスタ数: {k_opt}')

# --- 本番クラスタリング ---
affinity = pairwise_kernels(features, metric='rbf', gamma=0.5)
D = np.diag(affinity.sum(axis=1))
L = D - affinity
vals, vecs = eigh(L, D)
# 固有ベクトル数はk_opt-1本だが、2次元以上になるようにk_opt+1まで取る
X_spec = vecs[:, 1:(k_opt+1)]
labels = KMeans(n_clusters=k_opt, random_state=0).fit_predict(X_spec)

print(f"X_spec shape: {X_spec.shape}")
print(f"Cluster label counts: {np.bincount(labels)}")

# k=2のときは必ず1次元可視化
if k_opt == 2:
    X_emb = X_spec[:, 0]
    plt.figure(figsize=(8,3))
    for i in range(k_opt):
        plt.scatter(np.arange(len(X_emb))[labels==i], X_emb[labels==i], label=f'Cluster {i}')
    plt.legend()
    plt.title('Spectral Clustering (1D, k=2)')
    plt.savefig('spectral_clustering_1d.png')
    plt.show()
else:
    # 2次元以上ならt-SNE可視化
    if X_spec.shape[1] >= 2:
        try:
            X_emb = TSNE(n_components=2, random_state=0).fit_transform(X_spec)
            plt.figure(figsize=(8,6))
            for i in range(k_opt):
                plt.scatter(X_emb[labels==i,0], X_emb[labels==i,1], label=f'Cluster {i}')
            plt.legend()
            plt.title('Spectral Clustering of Heart Sounds (t-SNE)')
            plt.savefig('spectral_clustering_tsne.png')
            plt.show()
        except Exception as e:
            print(f"t-SNE失敗: {e}\n1次元可視化に切り替えます")
            X_emb = X_spec[:, 0]
            plt.figure(figsize=(8,3))
            for i in range(k_opt):
                plt.scatter(np.arange(len(X_emb))[labels==i], X_emb[labels==i], label=f'Cluster {i}')
            plt.legend()
            plt.title('Spectral Clustering (1D)')
            plt.savefig('spectral_clustering_1d.png')
            plt.show()
    else:
        # 1次元可視化
        X_emb = X_spec[:, 0]
        plt.figure(figsize=(8,3))
        for i in range(k_opt):
            plt.scatter(np.arange(len(X_emb))[labels==i], X_emb[labels==i], label=f'Cluster {i}')
        plt.legend()
        plt.title('Spectral Clustering (1D)')
        plt.savefig('spectral_clustering_1d.png')
        plt.show()

# クラスタごとの平均MFCCプロット
plt.figure(figsize=(10,4))
for i in range(k_opt):
    mean_mfcc = features[labels==i].mean(axis=0)
    plt.plot(mean_mfcc, label=f'Cluster {i}')
plt.legend()
plt.title('Cluster-wise Mean MFCC')
plt.savefig('cluster_mean_mfcc.png')
plt.show() 