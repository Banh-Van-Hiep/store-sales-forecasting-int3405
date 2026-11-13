from pathlib import Path
import numpy as np
import pandas as pd


def build_v1_dataset(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """
    Pipeline V1 dành cho Người 1:
    - Đọc & ghép các bảng dữ liệu thô (train, test, stores, oil, holidays_events)
    - Nội suy giá dầu cho ngày nghỉ/cuối tuần
    - Xử lý thông tin ngày lễ quốc gia (National holidays)
    - Tạo đặc trưng thời gian cơ bản
    - Cắt tập Validation chuẩn 16 ngày cuối cùng của tập train (2017-07-31 -> 2017-08-15)
    - Xuất ra định dạng .parquet nhẹ, tốc độ đọc nhanh cho Người 2 & Người 3
    """
    raw_path = Path(raw_dir)
    out_path = Path(processed_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("1. Đang đọc dữ liệu thô từ", raw_path)
    train = pd.read_csv(raw_path / "train.csv", parse_dates=["date"])
    test = pd.read_csv(raw_path / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(raw_path / "stores.csv")
    oil = pd.read_csv(raw_path / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(raw_path / "holidays_events.csv", parse_dates=["date"])

    train["is_test"] = 0
    test["is_test"] = 1
    test["sales"] = np.nan

    df = pd.concat([train, test], ignore_index=True)

    print("2. Đang xử lý và ghép bảng stores, oil, holidays...")
    df = df.merge(stores, on="store_nbr", how="left")

    full_dates = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")})
    oil_clean = full_dates.merge(oil, on="date", how="left")
    oil_clean["dcoilwtico"] = oil_clean["dcoilwtico"].interpolate(method="linear").bfill().ffill()
    df = df.merge(oil_clean, on="date", how="left")

    nat_holidays = holidays[(holidays["locale"] == "National") & (~holidays["transferred"])].copy()
    nat_holidays = nat_holidays.drop_duplicates(subset=["date"])[["date", "type"]].rename(
        columns={"type": "national_holiday_type"}
    )
    df = df.merge(nat_holidays, on="date", how="left")
    df["is_national_holiday"] = df["national_holiday_type"].notna().astype("int8")
    df["national_holiday_type"] = df["national_holiday_type"].fillna("None")

    print("3. Đang tạo đặc trưng thời gian cơ bản (V1)...")
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
    df["day_of_month"] = df["date"].dt.day.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["year"] = df["date"].dt.year.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_payday"] = ((df["day_of_month"] == 15) | (df["date"].dt.is_month_end)).astype("int8")

    cat_cols = ["family", "city", "state", "type", "national_holiday_type"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    df["store_nbr"] = df["store_nbr"].astype("int16")
    df["cluster"] = df["cluster"].astype("int16")
    df["onpromotion"] = df["onpromotion"].astype("int32")

    print("4. Đang cắt tập Train / Validation (16 ngày cuối) / Test...")
    test_df = df[df["is_test"] == 1].drop(columns=["is_test", "sales"]).reset_index(drop=True)
    full_train_df = df[df["is_test"] == 0].drop(columns=["is_test"]).reset_index(drop=True)

    max_train_date = full_train_df["date"].max()
    val_start_date = max_train_date - pd.Timedelta(days=15)

    train_split = full_train_df[full_train_df["date"] < val_start_date].reset_index(drop=True)
    val_split = full_train_df[full_train_df["date"] >= val_start_date].reset_index(drop=True)

    print(f"   - Train period: {train_split['date'].min().date()} -> {train_split['date'].max().date()} ({len(train_split):,} dòng)")
    print(f"   - Val period  : {val_split['date'].min().date()} -> {val_split['date'].max().date()} ({len(val_split):,} dòng)")
    print(f"   - Test period : {test_df['date'].min().date()} -> {test_df['date'].max().date()} ({len(test_df):,} dòng)")

    print("5. Đang lưu file .parquet ra", out_path)
    train_split.to_parquet(out_path / "train_v1.parquet", index=False)
    val_split.to_parquet(out_path / "val_v1.parquet", index=False)
    test_df.to_parquet(out_path / "test_v1.parquet", index=False)
    print("Hoàn tất tạo bộ dữ liệu V1!")


if __name__ == "__main__":
    build_v1_dataset()
