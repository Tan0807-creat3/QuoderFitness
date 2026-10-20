"""
Upload QoderFitness model len Kaggle Models (kagglehub).
Chay tren may ban (can dang nhap Kaggle truoc).
"""

import kagglehub
import os

# ==================== CẤU HÌNH ====================
LOCAL_MODEL_DIR = "models_upload"   # Thu mục chứa file model
MODEL_SLUG = "qoderfitness"         # Slug model trên Kaggle (thay đổi nếu muốn)
VARIATION_SLUG = "default"          # Slug biến thể (default, v1, v2...)
VERSION_NOTES = "Update 2026-10-20 - QoderFitness multi-task AI"
# ==================================================

def main():
    # Kiểm tra thư mục tồn tại
    if not os.path.isdir(LOCAL_MODEL_DIR):
        print(f"Không tìm thấy thư mục: {LOCAL_MODEL_DIR}")
        return

    handle = f"tannguyen13/{MODEL_SLUG}/pyTorch/{VARIATION_SLUG}"

    print(f"Đang upload model lên: {handle}")
    print(f"Thư mục nguồn: {os.path.abspath(LOCAL_MODEL_DIR)}")
    print(f"Ghi chú phiên bản: {VERSION_NOTES}")
    print("-" * 50)

    try:
        version = kagglehub.model_upload(
            handle=handle,
            local_model_dir=LOCAL_MODEL_DIR,
            version_notes=VERSION_NOTES,
        )
        print("-" * 50)
        print("Upload thành công!")
        print(f"Version: {version}")
    except Exception as e:
        print(f"Lỗi upload: {e}")
        print("\nHướng dẫn khắc phục:")
        print("1. Đăng nhập Kaggle: kagglehub.login()  hoac  kaggle login")
        print("2. Hoặc đặt file kaggle.json vào ~/.kaggle/kaggle.json")
        print("3. Đảm bảo MODEL_SLUG và VARIATION_SLUG hợp lệ trên Kaggle")


if __name__ == "__main__":
    main()
