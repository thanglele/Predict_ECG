import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pandas as pd
import os
import requests
import zipfile
from tqdm import tqdm
import pickle
from scipy import signal, stats

# Định nghĩa các hằng số
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
SAMPLING_RATE = 700  # Tần số lấy mẫu của PPG trong WESAD

def download_wesad(url, save_path='WESAD'):
    """
    Tải xuống và giải nén dataset WESAD
    """
        
    zip_path = os.path.join('WESAD.zip')
    
    print("Downloading WESAD dataset...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f:
        for data in tqdm(response.iter_content(chunk_size=8192),
                        total=total_size//8192,
                        desc="Downloading"):
            f.write(data)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)
    
    os.remove(zip_path)
    print("Dataset downloaded and extracted successfully!")

def extract_ppg_features(ppg_signal):
    """
    Trích xuất đặc trưng từ tín hiệu PPG
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        if len(ppg_signal) == 0:
            print("Lỗi: Tín hiệu PPG trống")
            return None
            
        features = {}
        
        # Đặc trưng thống kê cơ bản
        features['mean'] = float(np.mean(ppg_signal))
        features['std'] = float(np.std(ppg_signal))
        features['max'] = float(np.max(ppg_signal))
        features['min'] = float(np.min(ppg_signal))
        
        # Đặc trưng tần số
        try:
            freqs, psd = signal.welch(ppg_signal, fs=SAMPLING_RATE)
            features['peak_freq'] = float(freqs[np.argmax(psd)])
            features['power_total'] = float(np.sum(psd))
        except Exception as e:
            print(f"Lỗi khi tính đặc trưng tần số: {str(e)}")
            features['peak_freq'] = 0
            features['power_total'] = 0
        
        # Heart Rate Variability features
        try:
            peaks, _ = signal.find_peaks(ppg_signal, distance=SAMPLING_RATE//2)
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / SAMPLING_RATE
                features['hrv_mean'] = float(np.mean(rr_intervals))
                features['hrv_std'] = float(np.std(rr_intervals))
                features['hrv_rmssd'] = float(np.sqrt(np.mean(np.square(np.diff(rr_intervals)))))
            else:
                features['hrv_mean'] = 0
                features['hrv_std'] = 0
                features['hrv_rmssd'] = 0
        except Exception as e:
            print(f"Lỗi khi tính đặc trưng HRV: {str(e)}")
            features['hrv_mean'] = 0
            features['hrv_std'] = 0
            features['hrv_rmssd'] = 0
        
        # Kiểm tra giá trị NaN
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                print(f"Cảnh báo: Giá trị không hợp lệ trong {key}")
                features[key] = 0
                
        return features
        
    except Exception as e:
        print(f"Lỗi trong extract_ppg_features: {str(e)}")
        return None

def process_subject_data(subject_path):
    """
    Xử lý dữ liệu của một subject
    """
    try:
        with open(subject_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # In thông tin chi tiết để debug
        print(f"\nDebug info for {subject_path}:")
        print("Data keys:", data.keys())
        print("Signal keys:", data['signal'].keys())
        print("Wrist keys:", data['signal']['wrist'].keys())
        
        # Lấy dữ liệu PPG và labels
        ppg_data = np.array(data['signal']['wrist']['BVP']).flatten()  # Chuyển về 1D array
        labels = np.array(data['label'])
        
        print("PPG shape:", ppg_data.shape)
        print("Label shape:", labels.shape)
        print("Unique labels in data:", np.unique(labels))
        
        # Kiểm tra kích thước dữ liệu
        if len(ppg_data) != len(labels):
            print(f"Warning: PPG data length ({len(ppg_data)}) != labels length ({len(labels)})")
            # Cắt dữ liệu cho bằng nhau
            min_len = min(len(ppg_data), len(labels))
            ppg_data = ppg_data[:min_len]
            labels = labels[:min_len]
        
        features_list = []
        labels_list = []
        window_size = SAMPLING_RATE * 30  # 30 giây
        step_size = SAMPLING_RATE * 15    # 15 giây overlap
        
        for i in range(0, len(ppg_data) - window_size, step_size):
            window = ppg_data[i:i+window_size]
            window_labels = labels[i:i+window_size]
            unique_labels, counts = np.unique(window_labels, return_counts=True)
            window_label = unique_labels[np.argmax(counts)]
            
            # Chỉ xử lý baseline (1) và stress (2)
            if window_label in [1, 2]:
                features = extract_ppg_features(window)
                if features is not None:
                    features_list.append(features)
                    # Chuyển label từ [1,2] sang [0,1]
                    labels_list.append(1 if window_label == 2 else 0)
        
        print(f"Extracted {len(features_list)} valid windows from subject")
        print(f"Unique labels in processed data: {np.unique(labels_list)}")
        return features_list, labels_list
        
    except Exception as e:
        print(f"Error processing {subject_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], []

def prepare_wesad_data(base_path='WESAD'):
    """
    Chuẩn bị dữ liệu WESAD
    """
    all_features = []
    all_labels = []
    
    for subject in range(2, 18):  # WESAD có subject S2 đến S17
        subject_path = os.path.join(base_path, f'S{subject}', f'S{subject}.pkl')
        if os.path.exists(subject_path):
            print(f"\nProcessing subject S{subject}...")
            features, labels = process_subject_data(subject_path)
            if features and labels:  # Kiểm tra có dữ liệu không
                all_features.extend(features)
                all_labels.extend(labels)
                print(f"Added {len(features)} samples from subject S{subject}")
    
    if not all_features:
        raise ValueError("Không có dữ liệu được xử lý thành công!")
    
    # Chuyển list of dicts thành DataFrame
    features_df = pd.DataFrame(all_features)
    print("\nSummary:")
    print(f"Total samples: {len(features_df)}")
    print(f"Features: {features_df.columns.tolist()}")
    print(f"Class distribution: {np.bincount(all_labels)}")
    
    # Kiểm tra và xử lý dữ liệu không hợp lệ
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    if features_df.isnull().any().any():
        print("\nWarning: Found invalid values, filling with 0")
        features_df = features_df.fillna(0)
    
    # Chuẩn hóa features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_df)
    
    # Lưu dữ liệu đã xử lý
    np.save('wesad_ppg_features.npy', features_normalized)
    np.save('wesad_labels.npy', all_labels)
    
    with open('ppg_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return features_normalized, all_labels

def prepare_dataset():
    """
    Chuẩn bị toàn bộ dataset
    """
    data_path = 'WESAD'
    
    # Tải dataset nếu chưa có
    if not os.path.exists(data_path):
        download_wesad(WESAD_URL, data_path)
    
    try:
        # Xử lý dữ liệu PPG
        if not os.path.exists('wesad_ppg_features.npy'):
            features, labels = prepare_wesad_data(data_path)
            print(f"Processed data shape: {features.shape}")
            print(f"Number of samples: {len(labels)}")
            print(f"Class distribution: {np.bincount(labels)}")
        else:
            print("Đã tìm thấy dữ liệu đã xử lý.")
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu: {str(e)}")
        raise
    
    print("Dataset preparation completed!")

def load_ppg_data():
    """
    Load dữ liệu PPG đã xử lý
    """
    features = np.load('wesad_ppg_features.npy')
    labels = np.load('wesad_labels.npy')
    return features, labels

def create_ppg_model(input_shape, num_classes):
    """
    Tạo mô hình chỉ xử lý dữ liệu PPG
    """
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model():
    # Chuẩn bị dataset
    prepare_dataset()
    
    # Load PPG data
    ppg_features, ppg_labels = load_ppg_data()
    
    # Kiểm tra và in shape của dữ liệu
    print("\nData shapes before processing:")
    print(f"Features shape: {ppg_features.shape}")
    print(f"Labels shape: {ppg_labels.shape}")
    print(f"Unique labels: {np.unique(ppg_labels)}")
    
    # Đảm bảo labels là 1D array
    ppg_labels = ppg_labels.ravel()
    
    # One-hot encoding cho labels
    ppg_labels = to_categorical(ppg_labels, num_classes=2)
    
    print("\nData shapes after processing:")
    print(f"Features shape: {ppg_features.shape}")
    print(f"Labels shape: {ppg_labels.shape}")
    
    # Split data
    X_ppg_train, X_ppg_val, y_train, y_val = train_test_split(
        ppg_features, ppg_labels, test_size=0.2, random_state=42,
        stratify=ppg_labels  # Đảm bảo phân bố classes cân bằng
    )
    
    # In thông tin về tập train/val
    print("\nTraining/Validation split:")
    print(f"Training features shape: {X_ppg_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation features shape: {X_ppg_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    
    # Create and compile model for PPG only
    model = create_ppg_model(
        input_shape=(ppg_features.shape[1],),
        num_classes=2  # Stress vs Non-stress
    )
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    history = model.fit(
        X_ppg_train,
        y_train,
        validation_data=(X_ppg_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_ppg_val, y_val, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save model
    model.save('stress_detection_model.h5')
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()