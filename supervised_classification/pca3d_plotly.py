import numpy as np
import plotly.express as px

# npzファイルの読み込み
data = np.load('pca3d_result.npz', allow_pickle=True)
X_pca3 = data['X_pca3']
labels_num = data['labels_num']
class_names = data['class_names']

# ラベル名を数値から文字列に変換
display_labels = [class_names[i] for i in labels_num]

# Plotlyで3D散布図
fig = px.scatter_3d(
    x=X_pca3[:,0], y=X_pca3[:,1], z=X_pca3[:,2],
    color=display_labels,
    labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
    title='PCA of MFCC Features (3D, Plotly)'
)
fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.show() 