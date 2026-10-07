"""
features.py - โมดูลกลางสำหรับ Feature Engineering ของ FitMetrix AI

ใช้ร่วมกันทั้งตอน "เทรนโมเดล" (Colab / training/train.py) และตอน "รันแอป" (Streamlit)
ไฟล์ .pkl จะอ้างอิง class จากโมดูลนี้ (features.FeatureEngineer) ดังนั้น:
  - ต้องมีไฟล์ features.py อยู่ในที่ที่ Python import ได้ ตอนเรียก joblib.load()
  - ถ้าย้ายโมดูลนี้ไปไว้ที่อื่น (เช่น src/features.py) ต้องเทรนและเซฟโมเดลใหม่
    ด้วย import path เดียวกัน
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def calc_bmi(weight_kg, height_cm):
    """BMI = น้ำหนัก(kg) / ส่วนสูง(m)^2"""
    return weight_kg / ((height_cm / 100.0) ** 2)


def calc_bmr(weight_kg, height_cm, age, gender):
    """
    BMR สูตร Mifflin-St Jeor (kcal/วัน)
        ชาย  = 10*W + 6.25*H - 5*A + 5
        หญิง = 10*W + 6.25*H - 5*A - 161
    gender รับ 'male' / 'female' (ไม่สนตัวพิมพ์เล็ก-ใหญ่) รองรับทั้งค่าเดี่ยวและ pandas Series
    ถ้า gender ไม่รู้จัก จะได้ NaN (ให้ SimpleImputer ใน Pipeline จัดการต่อ)
    """
    import pandas as pd

    g = pd.Series(gender).astype(str).str.strip().str.lower()
    offset = np.select([g == "male", g == "female"], [5.0, -161.0], default=np.nan)
    result = 10 * np.asarray(weight_kg) + 6.25 * np.asarray(height_cm) - 5 * np.asarray(age) + offset
    return result if np.ndim(weight_kg) > 0 else float(result[0])


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """เพิ่มคอลัมน์ BMI และ BMR ให้ DataFrame (ต้องมี Gender, Age, Height, Weight)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["BMI"] = calc_bmi(X["Weight"], X["Height"])
        X["BMR"] = calc_bmr(X["Weight"].values, X["Height"].values, X["Age"].values, X["Gender"])
        return X
