# Phân loại ảnh thật/giả sử dụng PyTorch

Dự án này sử dụng PyTorch để xây dựng mô hình học sâu phân loại ảnh thật và ảnh do AI tạo ra. Mô hình sử dụng kiến trúc CNN (Convolutional Neural Network) để phân loại.

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Windows/Linux/MacOS

## Cài đặt

1. Cài đặt Python (nếu chưa có):
   - Tải Python từ [python.org](https://www.python.org/downloads/)
   - Chọn phiên bản phù hợp với hệ điều hành
   - Trong quá trình cài đặt, nhớ tích vào "Add Python to PATH"

2. Cài đặt các thư viện cần thiết:
```bash
pip install torch torchvision pillow numpy scikit-learn matplotlib streamlit
```

## Cấu trúc thư mục

```
PythonProject4/
├── data/
│   ├── real/     # Chứa ảnh thật
│   └── fake/     # Chứa ảnh AI-generated
├── saved_model/  # Thư mục lưu model đã train
├── model.py      # Định nghĩa mô hình CNN
├── data_loader.py# Xử lý và tải dữ liệu
├── train.py      # Script huấn luyện mô hình
└── app.py        # Ứng dụng web Streamlit
```

## Cách sử dụng

### 1. Chuẩn bị dữ liệu
- Tạo thư mục `data` trong thư mục dự án
- Trong thư mục `data`, tạo 2 thư mục con:
  - `real`: chứa các ảnh thật
  - `fake`: chứa các ảnh do AI tạo ra

### 2. Huấn luyện mô hình
Mở Command Prompt (CMD) và chạy các lệnh sau:
```bash
cd đường_dẫn_đến_thư_mục_PythonProject4
python train.py
```

### 3. Chạy ứng dụng
Mở Command Prompt (CMD) và chạy các lệnh sau:
```bash
cd đường_dẫn_đến_thư_mục_PythonProject4
python -m streamlit run app.py
```

Sau khi chạy:
1. Trình duyệt web sẽ tự động mở tại địa chỉ: http://localhost:8501
2. Tải lên ảnh cần kiểm tra
3. Xem kết quả phân loại

## Xử lý lỗi thường gặp

1. Lỗi "Import không tìm thấy":
   - Kiểm tra đã cài đặt đầy đủ thư viện chưa
   - Chạy lại lệnh cài đặt thư viện

2. Lỗi "Không tìm thấy model":
   - Kiểm tra đã chạy huấn luyện mô hình chưa
   - Kiểm tra thư mục `saved_model` đã được tạo chưa

3. Lỗi "Không tìm thấy dữ liệu":
   - Kiểm tra cấu trúc thư mục `data`
   - Đảm bảo có ảnh trong thư mục `real` và `fake`

## Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra phần "Xử lý lỗi thường gặp"
2. Tạo issue mới với mô tả chi tiết vấn đề

## Giấy phép

MIT License 