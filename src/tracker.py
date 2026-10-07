"""
tracker.py - Phase 3: Real-time Calorie Tracker ของ FitMetrix AI

หน้าที่:
  1) ตรวจสอบ input (ค่าที่เป็นไปไม่ได้ทางสรีรวิทยา -> ValidationError)
  2) เทียบ input กับ "ช่วงข้อมูลที่ใช้เทรน" (model_meta.json)
  3) ทำนายแคลอรี:
       - อยู่ในช่วงข้อมูลฝึก       -> ใช้โมเดล ML (confidence = "high")
       - อยู่นอกช่วง + policy="fallback"   -> ใช้สูตร Keytel et al. (confidence = "medium")
       - อยู่นอกช่วง + policy="model_warn" -> ใช้โมเดล ML แต่เตือนว่าไม่น่าเชื่อถือ (confidence = "low")
  4) จำแนก Heart Rate Zone แบบ rule-based (HRmax สูตร Tanaka)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # ให้ joblib.load() หาโมดูล features.py เจอ

MODEL_PATH = ROOT / "models" / "fitmetrix_model.pkl"
META_PATH = ROOT / "models" / "model_meta.json"

# ขอบเขตที่ "เป็นไปได้ทางสรีรวิทยา" ถ้าเกินถือว่ากรอกผิด -> ValidationError
HARD_LIMITS = {
    "Age": (10, 100),
    "Height": (100, 250),
    "Weight": (30, 250),
    "Duration": (1, 600),
    "Heart_Rate": (40, 230),
}
LABELS_TH = {
    "Age": "อายุ (ปี)",
    "Height": "ส่วนสูง (ซม.)",
    "Weight": "น้ำหนัก (กก.)",
    "Duration": "เวลาออกกำลังกาย (นาที)",
    "Heart_Rate": "ชีพจร (ครั้ง/นาที)",
}
VALID_POLICIES = ("fallback", "model_warn")


class ValidationError(ValueError):
    """input ไม่ถูกต้อง เก็บข้อความทุกข้อผิดพลาดไว้ใน .errors"""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


# ----------------------------- Data classes ---------------------------------
@dataclass
class UserProfile:
    gender: str          # "male" / "female"
    age: float
    height_cm: float
    weight_kg: float


@dataclass
class Workout:
    duration_min: float
    heart_rate: float                 # ชีพจรเฉลี่ยตลอดการออกกำลังกาย
    activity: str = "cardio"          # "cardio" / "strength" / "other"


@dataclass
class TrackerResult:
    calories: float
    method: str                       # "ml_model" | "keytel_formula"
    confidence: str                   # "high" | "medium" | "low"
    hr_max: float
    zone: dict
    warnings: list = field(default_factory=list)
    out_of_range: dict = field(default_factory=dict)   # {feature: {"value":..,"min":..,"max":..}}

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------- Heart Rate Zones -----------------------------
# (หมายเลข, ชื่อ, %HRmax ต่ำสุด, %HRmax สูงสุด, คำอธิบาย)
ZONES = [
    (1, "Zone 1 - Very Light", 0.50, 0.60, "วอร์มอัพ / ฟื้นตัว"),
    (2, "Zone 2 - Light", 0.60, 0.70, "เบา สร้างความอึด เน้นใช้ไขมัน"),
    (3, "Zone 3 - Moderate", 0.70, 0.80, "ปานกลาง พัฒนาระบบแอโรบิก"),
    (4, "Zone 4 - Hard", 0.80, 0.90, "หนัก เพิ่มสมรรถภาพ"),
    (5, "Zone 5 - Maximum", 0.90, 1.00, "หนักสุด ช่วงสั้นๆ เท่านั้น"),
]


def calc_hr_max(age: float) -> float:
    """HRmax สูตร Tanaka = 208 - 0.7 x อายุ"""
    return 208.0 - 0.7 * age


def classify_zone(heart_rate: float, age: float) -> dict:
    hr_max = calc_hr_max(age)
    pct = heart_rate / hr_max
    if pct < 0.50:
        return {"zone": 0, "name": "Below Zone 1", "description": "ต่ำกว่าช่วงออกกำลังกาย", "pct_hrmax": round(pct * 100, 1)}
    chosen = ZONES[-1]
    for z in ZONES:
        if pct < z[3]:
            chosen = z
            break
    return {"zone": chosen[0], "name": chosen[1], "description": chosen[4], "pct_hrmax": round(pct * 100, 1)}


def zone_table(age: float) -> list[dict]:
    """ตารางช่วงชีพจรของแต่ละโซนสำหรับอายุนี้ (ไว้แสดงใน UI)"""
    hr_max = calc_hr_max(age)
    return [
        {"Zone": z[1], "%HRmax": f"{int(z[2]*100)}-{int(z[3]*100)}%",
         "Heart Rate (bpm)": f"{round(hr_max*z[2])}-{round(hr_max*z[3])}", "ความหมาย": z[4]}
        for z in ZONES
    ]


# ----------------------------- สูตรสำรอง (Keytel et al., 2005) ---------------
def keytel_calories(gender: str, age: float, weight_kg: float, heart_rate: float, duration_min: float) -> float:
    """
    ประมาณแคลอรีที่เผาผลาญจากชีพจร (kcal) สูตร Keytel et al. (2005)
        ชาย  : (-55.0969 + 0.6309*HR + 0.1988*W + 0.2017*A) / 4.184  [kcal/นาที]
        หญิง : (-20.4022 + 0.4472*HR - 0.1263*W + 0.0740*A) / 4.184  [kcal/นาที]
    เป็นสูตรประมาณสำหรับคาร์ดิโอ ไม่ใช่ค่าที่วัดจริง ถ้าติดลบ (ชีพจรต่ำมาก) ปรับเป็น 0
    """
    if gender == "male":
        per_min = (-55.0969 + 0.6309 * heart_rate + 0.1988 * weight_kg + 0.2017 * age) / 4.184
    else:
        per_min = (-20.4022 + 0.4472 * heart_rate - 0.1263 * weight_kg + 0.0740 * age) / 4.184
    return max(0.0, per_min * duration_min)


# ----------------------------- Tracker --------------------------------------
class CalorieTracker:
    def __init__(self, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH, policy: str = "fallback"):
        if policy not in VALID_POLICIES:
            raise ValueError(f"policy ต้องเป็นหนึ่งใน {VALID_POLICIES}")
        self.policy = policy

        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.model = joblib.load(model_path)

        self.load_warnings: list[str] = []
        trained_ver = self.meta.get("versions", {}).get("scikit-learn")
        if trained_ver and trained_ver != sklearn.__version__:
            self.load_warnings.append(
                f"เวอร์ชัน scikit-learn ไม่ตรงกัน (เทรน {trained_ver} / เครื่องนี้ {sklearn.__version__}) "
                f"แนะนำ pip install scikit-learn=={trained_ver}"
            )

    # ---- 1) ตรวจ input ----
    @staticmethod
    def _normalize_gender(gender: str) -> str:
        g = str(gender).strip().lower()
        return {"m": "male", "ชาย": "male", "f": "female", "หญิง": "female"}.get(g, g)

    def _validate(self, profile: UserProfile, workout: Workout) -> tuple[str, dict]:
        errors = []
        gender = self._normalize_gender(profile.gender)
        if gender not in ("male", "female"):
            errors.append("เพศต้องเป็น male หรือ female")

        values = {
            "Age": profile.age, "Height": profile.height_cm, "Weight": profile.weight_kg,
            "Duration": workout.duration_min, "Heart_Rate": workout.heart_rate,
        }
        for key, v in values.items():
            try:
                v = float(v)
                if v != v:  # NaN
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"{LABELS_TH[key]} ต้องเป็นตัวเลข")
                continue
            values[key] = v
            lo, hi = HARD_LIMITS[key]
            if not lo <= v <= hi:
                errors.append(f"{LABELS_TH[key]} = {v:g} อยู่นอกช่วงที่เป็นไปได้ ({lo}-{hi})")
        if errors:
            raise ValidationError(errors)
        return gender, values

    # ---- 2) เทียบกับช่วงข้อมูลฝึก ----
    def _check_training_range(self, values: dict) -> dict:
        out = {}
        for key, rng in self.meta["feature_ranges"].items():
            v = values[key]
            if not rng["min"] <= v <= rng["max"]:
                out[key] = {"value": v, "min": rng["min"], "max": rng["max"]}
        return out

    def _predict_model(self, gender: str, values: dict) -> float:
        row = {"Gender": gender, **values}
        X = pd.DataFrame([row])[self.meta["input_columns"]]
        return max(0.0, float(self.model.predict(X)[0]))

    # ---- 3) ทำนาย ----
    def predict(self, profile: UserProfile, workout: Workout) -> TrackerResult:
        gender, values = self._validate(profile, workout)
        out_of_range = self._check_training_range(values)
        warnings: list[str] = []

        if out_of_range:
            detail = ", ".join(
                f"{LABELS_TH[k]} = {d['value']:g} (ข้อมูลฝึกครอบคลุม {d['min']:g}-{d['max']:g})"
                for k, d in out_of_range.items()
            )
            if self.policy == "fallback":
                calories = keytel_calories(gender, values["Age"], values["Weight"], values["Heart_Rate"], values["Duration"])
                method, confidence = "keytel_formula", "medium"
                warnings.append(f"ค่าที่กรอกอยู่นอกช่วงที่โมเดลเคยเรียนรู้: {detail} จึงใช้สูตรมาตรฐานจากชีพจรแทนโมเดล ML")
            else:
                calories = self._predict_model(gender, values)
                method, confidence = "ml_model", "low"
                warnings.append(f"ค่าที่กรอกอยู่นอกช่วงที่โมเดลเคยเรียนรู้: {detail} ผลทำนายอาจคลาดเคลื่อนมาก")
        else:
            calories = self._predict_model(gender, values)
            method, confidence = "ml_model", "high"

        hr_max = calc_hr_max(values["Age"])
        if values["Heart_Rate"] > hr_max:
            warnings.append(f"ชีพจรสูงกว่า HRmax โดยประมาณ ({hr_max:.0f} bpm) ตรวจสอบค่าที่กรอก และหยุดพักหากรู้สึกไม่สบาย")
        if str(workout.activity).lower() == "strength":
            warnings.append("การประเมินจากชีพจรเหมาะกับคาร์ดิโอมากกว่า เวทเทรนนิ่งชีพจรไม่สะท้อนพลังงานที่ใช้ได้ดี ผลอาจคลาดเคลื่อน")

        return TrackerResult(
            calories=round(calories, 1),
            method=method,
            confidence=confidence,
            hr_max=round(hr_max, 1),
            zone=classify_zone(values["Heart_Rate"], values["Age"]),
            warnings=warnings,
            out_of_range=out_of_range,
        )


@lru_cache(maxsize=2)
def get_tracker(policy="model_warn") -> CalorieTracker:
    """โหลดโมเดลครั้งเดียวแล้วใช้ซ้ำ"""
    return CalorieTracker(policy=policy)