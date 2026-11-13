import numpy as np
from sklearn.metrics import root_mean_squared_error


def compute_rmsle(y_true, y_pred) -> float:
    """
    Tính chỉ số RMSLE (Root Mean Squared Logarithmic Error) chuẩn của cuộc thi.
    Tự động cắt các giá trị âm về 0 theo đúng quy định doanh số không âm.
    """
    y_true = np.clip(np.asarray(y_true, dtype=np.float64), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    return float(root_mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def to_log_target(y):
    """Biến đổi biến mục tiêu sang log(1 + y) để tối ưu trực tiếp RMSE trên không gian log."""
    y = np.clip(np.asarray(y, dtype=np.float64), 0, None)
    return np.log1p(y)


def from_log_target(y_log):
    """Chuyển dự đoán từ không gian log về doanh số thực tế exp(y_log) - 1 và chặn dưới tại 0."""
    return np.clip(np.expm1(np.asarray(y_log, dtype=np.float64)), 0, None)
