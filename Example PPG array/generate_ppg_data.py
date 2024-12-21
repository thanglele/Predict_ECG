import numpy as np
import pandas as pd

# Định nghĩa các tham số
sampling_rate = 700  # Tần số lấy mẫu (Hz)
duration = 30  # Thời gian ghi (giây)
num_samples = sampling_rate * duration  # Tổng số mẫu

# Tạo dữ liệu PPG giả lập
time = np.linspace(0, duration, num_samples)
# Giả lập tín hiệu PPG với một sóng sin và một số nhiễu ngẫu nhiên
ppg_signal = 0.5 * np.sin(2 * np.pi * 1.5 * time) + 0.05 * np.random.normal(size=num_samples)

# Tạo DataFrame và lưu vào file CSV
ppg_data = pd.DataFrame({'Time': time, 'BVP': ppg_signal})
ppg_data.to_csv('simulated_ppg_data.csv', index=False)

print("Mẫu dữ liệu PPG đã được tạo và lưu vào 'simulated_ppg_data.csv'.") 