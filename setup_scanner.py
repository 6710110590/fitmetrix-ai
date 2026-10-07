"""
setup_scanner.py - ตรวจสอบและเตรียมระบบ Scanner (รันครั้งเดียวก่อนใช้งาน)

    python setup_scanner.py

ทำ 3 อย่าง: ตรวจ mediapipe -> ดาวน์โหลดไฟล์โมเดลท่าทางถ้ายังไม่มี -> ทดลองโหลดและตรวจจับบนภาพว่าง
"""
import platform
import sys

import numpy as np

from src.scanner import MODEL_PATH, PoseDetector, ScannerSetupError, download_model

print(f"Python {platform.python_version()}")

try:
    import mediapipe as mp
    print(f"mediapipe {getattr(mp, '__version__', '(ไม่ทราบเวอร์ชัน)')} : พร้อมใช้งาน")
except ImportError:
    sys.exit("ไม่พบ mediapipe -> รัน: python -m pip install mediapipe")

try:
    if MODEL_PATH.exists():
        print(f"พบไฟล์โมเดลแล้ว: {MODEL_PATH}")
    else:
        print("กำลังดาวน์โหลดโมเดลท่าทาง (ประมาณ 6 MB)...")
        download_model()
        print(f"ดาวน์โหลดสำเร็จ: {MODEL_PATH}")

    detector = PoseDetector()
    result = detector.detect(np.zeros((480, 360, 3), dtype=np.uint8))
    detector.close()
    print("ทดสอบตรวจจับบนภาพว่าง:", "ไม่พบคน (ถูกต้อง)" if result is None else "พบคน (ผิดปกติ)")
    print("\nพร้อมใช้งาน เปิดแอปด้วย: streamlit run app.py แล้วเลือกหน้า Profile Scanner")
except ScannerSetupError as e:
    sys.exit(f"ตั้งค่าไม่สำเร็จ: {e}")