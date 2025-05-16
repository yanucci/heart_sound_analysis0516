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

def load_and_preprocess_audio(file_path, target_sr=5000, duration=1.0):
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = librosa.util.normalize(audio)
    if len(audio) > target_sr:
        audio = audio[:target_sr]
    else:
        audio = np.pad(audio, (0, target_sr - len(audio)))
    return audio

def create_spectrogram(audio, sr=5000):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=64,
        fmax=5000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    return mel_spec_norm

def create_stft_spectrogram(audio, sr=5000, n_fft=256, hop_length=64):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spec = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    max_bin = np.where(freqs <= 5000)[0][-1] + 1
    spec = spec[:max_bin, :]
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    return spec_norm

def create_dataset(data_dir):
    X = []
    y = []
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    for file_path in wav_files:
        subject_id = os.path.basename(file_path).split('_')[0]
        audio = load_and_preprocess_audio(file_path)
        spec = create_spectrogram(audio)
        X.append(spec)
        y.append(subject_id)
    X = np.array(X)
    y = np.array(y)
    print(f"スペクトログラムshape: {X.shape}")
    return X, y

def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.convert_to_tensor(img_array[None, ...], dtype=tf.float32))
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    # デバッグ出力
    print(f"[GradCAM] predictions: {predictions.numpy()}, class_idx: {class_idx.numpy()}")
    print(f"[GradCAM] heatmap before ReLU: min={heatmap.min()}, max={heatmap.max()}, mean={heatmap.mean()}")
    heatmap = np.maximum(heatmap, 0)
    print(f"[GradCAM] heatmap after ReLU: min={heatmap.min()}, max={heatmap.max()}, mean={heatmap.mean()}")
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap) + 1e-8
    print(f"[GradCAM] heatmap after norm: min={heatmap.min()}, max={heatmap.max()}, mean={heatmap.mean()}")
    return heatmap

def visualize_gradcam_on_stft(model, wav_path, layer_name, sr=5000, n_fft=256, hop_length=64, save_path=None):
    audio = load_and_preprocess_audio(wav_path, target_sr=sr)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)
    spec = np.abs(D)
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    input_shape = model.input.shape[1:3]
    spec_resized = tf.image.resize(spec_norm[..., np.newaxis], input_shape).numpy()
    heatmap = grad_cam(model, spec_resized, layer_name)
    print(f"{os.path.basename(wav_path)}: heatmap min={heatmap.min()}, max={heatmap.max()}")
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], S_db.shape).numpy().squeeze()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', fmax=5000)
    plt.title('Spectrogram (dB)')
    plt.colorbar(format='%+2.0f dB')
    plt.subplot(1, 2, 2)
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', fmax=5000)
    plt.imshow(heatmap_resized, cmap='jet', alpha=1.0, aspect='auto', extent=[times[0], times[-1], freqs[0], freqs[-1]], origin='lower', vmin=0, vmax=1)
    plt.title('Spectrogram + Grad-CAM')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        heatmap_only_path = save_path.replace('.png', '_heatmap.png')
        plt.figure(figsize=(6, 5))
        plt.imshow(heatmap_resized, cmap='jet', aspect='auto', extent=[times[0], times[-1], freqs[0], freqs[-1]], origin='lower', vmin=0, vmax=1)
        plt.title('Grad-CAM Heatmap Only')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(heatmap_only_path)
        plt.close()
    else:
        plt.savefig('gradcam_specshow.png')
    plt.close()

def visualize_gradcam_on_stft_single(model, wav_path, layer_name, sr=5000, n_fft=256, hop_length=64, save_path=None):
    audio = load_and_preprocess_audio(wav_path, target_sr=sr)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)
    spec = np.abs(D)
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    input_shape = model.input.shape[1:3]
    spec_resized = tf.image.resize(spec_norm[..., np.newaxis], input_shape).numpy()
    heatmap = grad_cam(model, spec_resized, layer_name)
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], S_db.shape).numpy().squeeze()
    heatmap_mask = heatmap_resized ** 0.3
    import matplotlib as mpl
    plt.figure(figsize=(8, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', fmax=5000, cmap='gray')
    plt.title('Spectrogram + Grad-CAM')
    plt.colorbar(format='%+2.0f dB', label='dB')
    plt.imshow(heatmap_mask, cmap='jet', alpha=0.8, aspect='auto', extent=[times[0], times[-1], freqs[0], freqs[-1]], origin='lower', vmin=0, vmax=1)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap='jet'), ax=plt.gca(), orientation='vertical', pad=0.02)
    cbar.set_label('Grad-CAM')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig('gradcam_overlay_strong.png')
    plt.close()

def visualize_gradcam_on_stft_masked(model, wav_path, layer_name, sr=5000, n_fft=256, hop_length=64, save_path=None, threshold=0.5):
    audio = load_and_preprocess_audio(wav_path, target_sr=sr)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)
    spec = np.abs(D)
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    input_shape = model.input.shape[1:3]
    spec_resized = tf.image.resize(spec_norm[..., np.newaxis], input_shape).numpy()
    heatmap = grad_cam(model, spec_resized, layer_name)
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], S_db.shape).numpy().squeeze()
    mask = heatmap_resized > threshold
    import matplotlib as mpl
    plt.figure(figsize=(8, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', fmax=5000, cmap='gray')
    plt.title('Spectrogram + Grad-CAM (masked)')
    plt.colorbar(format='%+2.0f dB', label='dB')
    plt.imshow(np.ma.masked_where(~mask, heatmap_resized), cmap='jet', alpha=0.8, aspect='auto', extent=[times[0], times[-1], freqs[0], freqs[-1]], origin='lower', vmin=0, vmax=1)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap='jet'), ax=plt.gca(), orientation='vertical', pad=0.02)
    cbar.set_label('Grad-CAM')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig('gradcam_overlay_masked.png')
    plt.close()

def batch_gradcam_masked_on_folder(model, folder_path, layer_name, out_folder, threshold=0.5):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    for wav_file in wav_files:
        wav_path = os.path.join(folder_path, wav_file)
        save_path = os.path.join(out_folder, f'gradcam_masked_{os.path.splitext(wav_file)[0]}.png')
        visualize_gradcam_on_stft_masked(model, wav_path, layer_name, save_path=save_path, threshold=threshold)

def save_gradcam_heatmap_only(model, wav_path, layer_name, sr=5000, n_fft=256, hop_length=64, save_path=None, stats_csv_path='gradcam_heatmap_stats.csv'):
    audio = load_and_preprocess_audio(wav_path, target_sr=sr)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)
    spec = np.abs(D)
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    input_shape = model.input.shape[1:3]
    spec_resized = tf.image.resize(spec_norm[..., np.newaxis], input_shape).numpy()
    heatmap = grad_cam(model, spec_resized, layer_name)
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], S_db.shape).numpy().squeeze()
    stats = {
        'file': os.path.basename(wav_path),
        'min': float(heatmap_resized.min()),
        'max': float(heatmap_resized.max()),
        'mean': float(heatmap_resized.mean()),
        'std': float(heatmap_resized.std())
    }
    print(f"{stats['file']}: heatmap min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    write_header = not os.path.exists(stats_csv_path)
    with open(stats_csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['file', 'min', 'max', 'mean', 'std'])
        if write_header:
            writer.writeheader()
        writer.writerow(stats)
    import matplotlib as mpl
    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap_resized, cmap='jet', aspect='auto', extent=[times[0], times[-1], freqs[0], freqs[-1]], origin='lower', vmin=0, vmax=1)
    plt.title('Grad-CAM Heatmap Only')
    plt.colorbar()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig('gradcam_heatmap_only.png')
    plt.close()

def batch_gradcam_heatmap_only_on_folder(model, folder_path, layer_name, out_folder, stats_csv_path='gradcam_heatmap_stats.csv'):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    if os.path.exists(stats_csv_path):
        os.remove(stats_csv_path)
    for wav_file in wav_files:
        wav_path = os.path.join(folder_path, wav_file)
        save_path = os.path.join(out_folder, f'gradcam_heatmap_{os.path.splitext(wav_file)[0]}.png')
        save_gradcam_heatmap_only(model, wav_path, layer_name, save_path=save_path, stats_csv_path=stats_csv_path)

def main():
    data_dir = "/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments"
    X, y = create_dataset(data_dir)
    X = X[..., np.newaxis]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    model = create_model(
        input_shape=(X.shape[1], X.shape[2], 1),
        num_classes=len(le.classes_)
    )
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    model.save('heart_sound_classifier.h5')
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
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    main()
    model = tf.keras.models.load_model('heart_sound_classifier.h5')
    layer_name = 'conv1'
    folder_path = '/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments'
    wav_path = '/Users/yusuke.yano/python_practice/heart_sound_analysis/data/segments/haradaR2LSB_segment_21.wav'
    visualize_gradcam_on_stft_single(model, wav_path, layer_name, save_path='gradcam_overlay_strong.png')
    out_folder = '/Users/yusuke.yano/python_practice/heart_sound_analysis/gradcam練習/gradcam_masked'
    batch_gradcam_masked_on_folder(model, folder_path, layer_name, out_folder, threshold=0.5)
    out_folder_heatmap = '/Users/yusuke.yano/python_practice/heart_sound_analysis/gradcam練習/gradcam_heatmap_only'
    batch_gradcam_heatmap_only_on_folder(model, folder_path, layer_name, out_folder_heatmap) 