import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy import signal

# Định nghĩa các hằng số
IMG_SIZE = 48
SAMPLING_RATE = 700  # Tần số lấy mẫu của PPG

# Tải mô hình đã train
model = load_model('combined_stress_detection_model.h5')

# Khởi tạo camera
cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định

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
                
        # Chuyển đổi dictionary thành mảng NumPy
        features_array = np.array(list(features.values()), dtype=np.float32)
    
        return features_array
        
    except Exception as e:
        print(f"Lỗi trong extract_ppg_features: {str(e)}")
        return None

# Hàm để lấy dữ liệu PPG từ cảm biến
def get_ppg_data():
    
    ## ĐOẠN NÀY CODE CONNECT VỚI CẢM BIẾN EEG ##
    
    # Tạm thời đang trả về 30 giây dữ liệu giả
    return np.random.rand(SAMPLING_RATE * 30)

while True:
    # Đọc dữ liệu từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc từ camera.")
        break
    
    # Tiền xử lý ảnh
    face_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (IMG_SIZE, IMG_SIZE))
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=[0, -1])  # Thêm batch dimension và channel dimension
    
    # Lấy dữ liệu PPG
    ppg_signal = get_ppg_data()
    ppg_features = extract_ppg_features(ppg_signal)  # Trích xuất các đặc trưng từ PPG
    
    # Chuyển đổi kiểu dữ liệu của ppg_features
    ppg_features = np.expand_dims(ppg_features, axis=0)  # Thêm batch dimension
    
    # Dự đoán mức độ stress
    prediction = model.predict([face_image, ppg_features])
    stress_level = np.argmax(prediction[0])
    confidence = prediction[0][stress_level]
    
    # Hiển thị kết quả
    if stress_level == 1:
        label = "Stressed"
    else:
        label = "Not Stressed"
    
    cv2.putText(frame, f"{label} (Confidence: {confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Stress Detection", frame)
    
    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng tất cả các cửa sổ
cap.release()
cv2.destroyAllWindows() 