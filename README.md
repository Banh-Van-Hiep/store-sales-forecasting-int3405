# Store Sales - Time Series Forecasting (INT3405)

Dự án Bài tập lớn môn **Học Máy (INT3405)** - Dự báo doanh số bán hàng cho 1.782 chuỗi thời gian (54 cửa hàng × 33 nhóm sản phẩm) trong 16 ngày tiếp theo trên Kaggle.

## Cấu trúc thư mục (`D:\ML`)
```text
├── data/
│   ├── raw/          # Chứa file dữ liệu thô tải từ Kaggle (không push lên Git)
│   └── processed/    # Chứa dữ liệu sạch V1 (.parquet) dùng chung cho cả nhóm
├── notebooks/
│   └── 01_eda_and_preprocessing_v1.ipynb  # Pipeline xử lý & cắt Validation (Người 1)
├── src/
│   ├── __init__.py
│   ├── metrics.py    # Hàm tính RMSLE chuẩn Kaggle dùng chung
│   └── preprocess.py # Script chạy nhanh tạo dữ liệu V1
├── submissions/      # Thư mục lưu file submission.csv trước khi nộp
├── .gitignore
└── requirements.txt
```

## Hướng dẫn chạy nhanh (Bản V1 trên ổ D)
1. Tạo môi trường ảo và cài đặt thư viện:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\pip install --cache-dir D:\ML\.pip_cache -r requirements.txt
   ```
2. Đặt các file `.csv` từ Kaggle vào `data/raw/` và chạy:
   ```powershell
   .\.venv\Scripts\python src/preprocess.py
   ```
