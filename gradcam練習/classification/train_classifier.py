"""
心音データの分類モデル学習と評価
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

def load_and_preprocess_data(feature_file):
    """
    特徴量データの読み込みと前処理
    """
    # データの読み込み
    df = pd.read_csv(feature_file)
    
    # 特徴量とラベルの分離
    X = df.drop(['file_name', 'dataset_name'], axis=1)
    y = df['dataset_name']
    
    # ラベルのエンコーディング
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # データの標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, le.classes_, X.columns

def plot_learning_curve(estimator, X, y, title):
    """
    学習曲線のプロット
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1)
    
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    
    return plt

def plot_feature_importance(model, feature_names):
    """
    特徴量の重要度の可視化
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.tight_layout()
    
    return plt

def plot_pca_visualization(X, y, class_names):
    """
    PCAによる特徴量の可視化
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   label=class_name, alpha=0.7)
    
    plt.title('PCA Visualization of Heart Sound Features')
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    
    return plt

def main():
    # 出力ディレクトリの設定
    output_dir = "results/visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # データの読み込みと前処理
    feature_file = "data/heart_sound_features.csv"
    X, y, class_names, feature_names = load_and_preprocess_data(feature_file)
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # モデルの学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 予測と評価
    y_pred = model.predict(X_test)
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 学習曲線の描画
    plt_learning = plot_learning_curve(
        model, X, y, 
        'Learning Curves (Random Forest)'
    )
    plt_learning.savefig(os.path.join(output_dir, 'learning_curve.png'))
    plt_learning.close()
    
    # 特徴量の重要度の可視化
    plt_importance = plot_feature_importance(model, feature_names)
    plt_importance.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt_importance.close()
    
    # PCAによる可視化
    plt_pca = plot_pca_visualization(X, y, class_names)
    plt_pca.savefig(os.path.join(output_dir, 'pca_visualization.png'))
    plt_pca.close()
    
    # 混同行列の可視化
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    print("\n可視化結果を以下のディレクトリに保存しました：")
    print(f"- {output_dir}/")
    print("  - learning_curve.png")
    print("  - feature_importance.png")
    print("  - pca_visualization.png")
    print("  - confusion_matrix.png")

if __name__ == "__main__":
    main() 