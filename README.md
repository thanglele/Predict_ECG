## Predict_PPG
====================================================
# Hướng dẫn cài đặt

1. Chạy lệnh cài thư viện: pip install -r requirements.txt
2. Nếu không có file h5, pkl, npy thì chạy các lệnh theo thứ tự sau:
2.1: python PrepareDataset.py
2.2: python MixModelCreate.py
3. Chạy lệnh để chạy thử:
python TestWorking.py

====================================================
Code cần bổ sung thêm ở file TestWorking.py, phần hàm lấy dữ liệu từ cảm biến
Dữ liệu từ cảm biến có thể xem mẫu ở thư mục Example PPG array để xử lý

lele