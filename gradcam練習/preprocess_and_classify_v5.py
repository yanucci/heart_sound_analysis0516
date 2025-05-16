# preprocess_and_classify_v5.py
# v4をベースに、YAMNetを使用した転移学習の実装

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
import glob
import csv
from scipy.signal import resample
import tensorflow_hub as hub

def load_and_preprocess_audio(file_path, target_sr=16000, target_length=512):
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = librosa.util.normalize(audio)
    # リスケーリングで長さを統一
    audio = resample(audio, target_length)
    return audio.astype(np.float32)

def create_dataset(data_dir, target_length=512):
    X = []
    y = []
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    for file_path in wav_files:
        subject_id = os.path.basename(file_path).split('_')[0]
        audio = load_and_preprocess_audio(file_path, target_length=target_length)
        X.append(audio)
        y.append(subject_id)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"音声データshape: {X.shape}")
    return X, y

class YAMNetFeatureExtractor(layers.Layer):
    def __init__(self, output_layer_name=None, **kwargs):
        super(YAMNetFeatureExtractor, self).__init__(**kwargs)
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.output_layer_name = output_layer_name  # 例: 'yamnet/conv_1'

    def call(self, inputs):
        # inputs: (batch, 512)
        def extract_fn(waveform):
            # waveform: (512,)
            scores, embeddings, spectrogram = self.yamnet_model(waveform)
            return embeddings[0]  # (1024,)
        embeddings = tf.map_fn(extract_fn, inputs, dtype=tf.float32)
        return embeddings  # (batch, 1024)

def create_model_with_yamnet(num_classes):
    inputs = Input(shape=(512,), dtype=tf.float32)
    x = YAMNetFeatureExtractor()(inputs)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='dense_2')(x)
    model = Model(inputs, outputs)
    return model

def get_yamnet_conv_output(waveform, layer_name='yamnet/conv_1'):
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    # yamnetはSavedModel形式なので、signaturesを使って中間層出力を取得
    # ただし、直接Kerasのように中間層出力を得るAPIはないため、
    # tf.functionでhookする必要がある
    # ここではconv_1層の出力を得る例
    # yamnet.signatures['serving_default']のinput/outputを調査
    # 公式のyamnet/model.pyを参考にする
    # ここでは簡易的にembeddingsのみ返す
    scores, embeddings, spectrogram = yamnet(waveform)
    # conv_1の出力は直接取得できないため、ここではembeddingsを返す
    # 本来はTensorFlow SavedModelのsignatureや内部構造を調査してhookする
    return embeddings

def grad_cam_yamnet(model, audio, layer_name):
    # audio: (512,)
    # YAMNetの中間層出力を取得
    conv_outputs = get_yamnet_conv_output(audio, layer_name=layer_name)  # (N, 1024)
    # ここではembeddingsを使う（本来はconv層出力が望ましい）
    # 勾配計算は省略し、単純な重要度可視化とする
    heatmap = np.mean(conv_outputs.numpy(), axis=-1)  # (N,)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def visualize_gradcam_on_audio(model, wav_path, layer_name, save_path=None):
    audio, sr = librosa.load(wav_path, sr=16000)
    audio = librosa.util.fix_length(audio, size=512)
    # heatmap取得
    heatmap = grad_cam_yamnet(model, audio, layer_name)
    # 可視化
    plt.figure(figsize=(10, 2))
    plt.plot(audio, label='Audio')
    plt.imshow(np.expand_dims(heatmap, axis=0), 
               aspect='auto', 
               cmap='jet', 
               alpha=0.5,
               extent=[0, len(audio), -1, 1])
    plt.colorbar()
    plt.title(f'Grad-CAM Visualization ({layer_name})')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def batch_gradcam_on_folder(model, folder_path, layer_name, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    for wav_file in wav_files:
        wav_path = os.path.join(folder_path, wav_file)
        save_path = os.path.join(out_folder, f'gradcam_yamnet_{os.path.splitext(wav_file)[0]}.png')
        visualize_gradcam_on_audio(model, wav_path, layer_name, save_path=save_path)

def main():
    data_dir = "/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments"
    X, y = create_dataset(data_dir)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    model = create_model_with_yamnet(num_classes=len(le.classes_))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    model.save('heart_sound_classifier_yamnet.h5', include_optimizer=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history_yamnet.png')
    plt.close()
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')

if __name__ == "__main__":
    main()
    model = tf.keras.models.load_model('heart_sound_classifier_yamnet.h5', custom_objects={'YAMNetFeatureExtractor': YAMNetFeatureExtractor})
    layer_name = 'dense_1'
    folder_path = '/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments'
    out_folder = '/Users/yusuke.yano/python_practice/heart_sound_analysis/gradcam練習/gradcam_yamnet'
    batch_gradcam_on_folder(model, folder_path, layer_name, out_folder) 