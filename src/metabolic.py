"""
metabolic.py - Phase 2: Metabolic Engine ของ FitMetrix AI

  BMR  (Mifflin-St Jeor)  -> TDEE (x ตัวคูณกิจกรรม) -> เป้าหมายแคลอรี/วัน -> Macros

ใช้สูตร BMR/BMI ชุดเดียวกับโมเดล ML (import จาก features.py) เพื่อไม่ให้สูตรสองที่ไม่ตรงกัน

กฎความปลอดภัยที่ใส่ไว้:
  - ให้คำแนะนำเฉพาะผู้ใหญ่ (18 ปีขึ้นไป)
  - แคลอรีเป้าหมายตอนลดไขมันต้องไม่ต่ำกว่า BMR และไม่ต่ำกว่า 1,500 (ชาย) / 1,200 (หญิง)
  - อัตราลดน้ำหนักไม่เกิน 1% ของน้ำหนักตัวต่อสัปดาห์
  - BMI < 18.5 จะไม่ตั้งเป้าลดไขมัน
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import calc_bmi, calc_bmr  # noqa: E402

ACTIVITY_FACTORS = {
    "sedentary": 1.2,       # แทบไม่ออกกำลังกาย (งานนั่งโต๊ะ)
    "light": 1.375,         # ออกกำลังกายเบา 1-3 วัน/สัปดาห์
    "moderate": 1.55,       # ปานกลาง 3-5 วัน/สัปดาห์
    "active": 1.725,        # หนัก 6-7 วัน/สัปดาห์
    "very_active": 1.9,     # หนักมากหรือใช้แรงงานหนัก
}
ACTIVITY_LABELS_TH = {
    "sedentary": "แทบไม่ออกกำลังกาย (งานนั่งโต๊ะ)",
    "light": "เบา (1-3 วัน/สัปดาห์)",
    "moderate": "ปานกลาง (3-5 วัน/สัปดาห์)",
    "active": "หนัก (6-7 วัน/สัปดาห์)",
    "very_active": "หนักมาก / งานใช้แรง",
}
GOALS = ("lose_fat", "build_muscle", "maintain")
GOAL_LABELS_TH = {"lose_fat": "ลดไขมัน", "build_muscle": "สร้างกล้ามเนื้อ", "maintain": "คงรูปร่าง"}
SOMATOTYPES = ("ectomorph", "mesomorph", "endomorph")

KCAL_PER_KG = 7700                 # พลังงานโดยประมาณของน้ำหนักตัว 1 กก.
MIN_CALORIES = {"male": 1500, "female": 1200}
MAX_WEEKLY_LOSS_PCT = 0.01         # ลดได้ไม่เกิน 1% ของน้ำหนักตัว/สัปดาห์
MIN_ADULT_AGE = 18

DEFAULT_ADJUST_PCT = {"lose_fat": 0.15, "build_muscle": 0.10, "maintain": 0.0}
ADJUST_LIMITS = {"lose_fat": (0.05, 0.25), "build_muscle": (0.05, 0.15)}

PROTEIN_G_PER_KG = {"lose_fat": 2.0, "build_muscle": 1.8, "maintain": 1.6}
# สัดส่วนตาม Somatotype เป็นแนวทางเชิงประมาณ (หลักฐานทางวิทยาศาสตร์รองรับจำกัด)
SOMATOTYPE_MACRO = {
    "ectomorph": {"fat_pct": 0.22, "protein_delta": 0.0},   # คาร์บสูงขึ้น
    "mesomorph": {"fat_pct": 0.27, "protein_delta": 0.0},   # สมดุล
    "endomorph": {"fat_pct": 0.30, "protein_delta": 0.2},   # คาร์บต่ำลง โปรตีนสูงขึ้น
}


class ProfileError(ValueError):
    """ข้อมูลโปรไฟล์ไม่ถูกต้อง เก็บทุกข้อความไว้ใน .errors"""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class NutritionPlan:
    gender: str
    goal: str
    activity_level: str
    somatotype: str
    bmi: float
    bmr: float
    tdee: float
    target_calories: float
    calorie_adjustment: float            # kcal/วัน เทียบกับ TDEE (ลบ = ลดแคลอรี)
    weekly_weight_change_kg: float       # ติดลบ = ลดน้ำหนัก
    weeks_to_goal: Optional[float]
    macros: dict
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def calc_tdee(bmr: float, activity_level: str) -> float:
    if activity_level not in ACTIVITY_FACTORS:
        raise ValueError(f"activity_level ต้องเป็นหนึ่งใน {list(ACTIVITY_FACTORS)}")
    return bmr * ACTIVITY_FACTORS[activity_level]


def calc_macros(target_calories: float, weight_kg: float, goal: str, somatotype: str) -> dict:
    cfg = SOMATOTYPE_MACRO[somatotype]
    protein_g_per_kg = PROTEIN_G_PER_KG[goal] + cfg["protein_delta"]
    protein_g = min(protein_g_per_kg * weight_kg, 0.35 * target_calories / 4)  # โปรตีนไม่เกิน 35% ของแคลอรี
    fat_g = cfg["fat_pct"] * target_calories / 9
    carbs_g = max(0.0, target_calories - protein_g * 4 - fat_g * 9) / 4

    grams = {"protein": round(protein_g), "carbs": round(carbs_g), "fat": round(fat_g)}
    kcal_per_g = {"protein": 4, "carbs": 4, "fat": 9}
    total_kcal = sum(grams[k] * kcal_per_g[k] for k in grams)
    macros = {
        k: {"grams": grams[k], "kcal": grams[k] * kcal_per_g[k],
            "pct": round(100 * grams[k] * kcal_per_g[k] / total_kcal, 1)}
        for k in grams
    }
    macros["protein_g_per_kg"] = round(grams["protein"] / weight_kg, 2)
    return macros


def build_nutrition_plan(
    gender: str,
    age: float,
    height_cm: float,
    weight_kg: float,
    target_weight_kg: float,
    activity_level: str,
    goal: str,
    somatotype: str = "mesomorph",
    adjust_pct: Optional[float] = None,
) -> NutritionPlan:
    """
    adjust_pct: สัดส่วนการลด (ลดไขมัน 0.05-0.25) หรือเพิ่ม (สร้างกล้ามเนื้อ 0.05-0.15) แคลอรีจาก TDEE
                ไม่ระบุ = ใช้ค่าเริ่มต้น (ลด 15% / เพิ่ม 10%)
    """
    # ---------- ตรวจ input ----------
    errors = []
    gender = str(gender).strip().lower()
    if gender not in ("male", "female"):
        errors.append("เพศต้องเป็น male หรือ female")
    if goal not in GOALS:
        errors.append(f"เป้าหมายต้องเป็นหนึ่งใน {list(GOALS)}")
    if activity_level not in ACTIVITY_FACTORS:
        errors.append(f"ระดับกิจกรรมต้องเป็นหนึ่งใน {list(ACTIVITY_FACTORS)}")
    if somatotype not in SOMATOTYPES:
        errors.append(f"Somatotype ต้องเป็นหนึ่งใน {list(SOMATOTYPES)}")
    for label, v, lo, hi in [
        ("อายุ", age, MIN_ADULT_AGE, 100), ("ส่วนสูง", height_cm, 100, 250),
        ("น้ำหนัก", weight_kg, 30, 250), ("น้ำหนักเป้าหมาย", target_weight_kg, 30, 250),
    ]:
        try:
            if not lo <= float(v) <= hi:
                errors.append(f"{label} ต้องอยู่ระหว่าง {lo}-{hi}"
                              + (" (ระบบนี้ให้คำแนะนำเฉพาะผู้ใหญ่)" if label == "อายุ" else ""))
        except (TypeError, ValueError):
            errors.append(f"{label} ต้องเป็นตัวเลข")
    if errors:
        raise ProfileError(errors)

    warnings: list[str] = []
    bmi = calc_bmi(weight_kg, height_cm)
    bmr = float(calc_bmr(weight_kg, height_cm, age, gender))
    tdee = calc_tdee(bmr, activity_level)

    # ---------- ปรับแคลอรีตามเป้าหมาย ----------
    pct = DEFAULT_ADJUST_PCT[goal] if adjust_pct is None else float(adjust_pct)
    if goal in ADJUST_LIMITS:
        lo, hi = ADJUST_LIMITS[goal]
        clamped = min(max(pct, lo), hi)
        if clamped != pct:
            warnings.append(f"ปรับสัดส่วนแคลอรีเป็น {clamped:.0%} เพื่ออยู่ในช่วงที่ปลอดภัย ({lo:.0%}-{hi:.0%})")
        pct = clamped

    if goal == "lose_fat":
        if bmi < 18.5:
            warnings.append(f"BMI ของคุณ {bmi:.1f} ต่ำกว่าเกณฑ์ (18.5) ระบบจึงไม่ตั้งเป้าลดแคลอรี แนะนำปรึกษาแพทย์หรือนักกำหนดอาหาร")
            target = tdee
        else:
            deficit = tdee * pct
            max_deficit = MAX_WEEKLY_LOSS_PCT * weight_kg * KCAL_PER_KG / 7
            if deficit > max_deficit:
                deficit = max_deficit
                warnings.append("ลดแคลอรีลงจากที่เลือกเล็กน้อย เพื่อให้น้ำหนักลดไม่เกิน 1% ของน้ำหนักตัวต่อสัปดาห์")
            target = tdee - deficit
            floor = min(max(bmr, MIN_CALORIES[gender]), tdee)
            if target < floor:
                target = floor
                warnings.append(f"ปรับแคลอรีเป้าหมายขึ้นเป็น {floor:.0f} kcal เพราะไม่ควรต่ำกว่า BMR และขั้นต่ำที่ปลอดภัย")
    elif goal == "build_muscle":
        target = tdee * (1 + pct)
    else:
        target = tdee

    adjustment = target - tdee
    weekly_change = adjustment * 7 / KCAL_PER_KG

    # ---------- เทียบกับน้ำหนักเป้าหมาย ----------
    diff = target_weight_kg - weight_kg
    weeks: Optional[float] = None
    if goal == "maintain":
        if abs(diff) > 0.5:
            warnings.append("เลือกคงรูปร่าง แต่น้ำหนักเป้าหมายต่างจากน้ำหนักปัจจุบัน ลองเปลี่ยนเป้าหมายเป็นลดไขมันหรือสร้างกล้ามเนื้อ")
    elif diff == 0:
        weeks = 0.0
    elif weekly_change != 0 and diff * weekly_change > 0:
        weeks = round(abs(diff / weekly_change), 1)
    elif weekly_change != 0:
        warnings.append("น้ำหนักเป้าหมายสวนทางกับเป้าหมายที่เลือก (เช่น เลือกลดไขมันแต่เป้าหมายน้ำหนักมากกว่าปัจจุบัน)")
    if calc_bmi(target_weight_kg, height_cm) < 18.5:
        warnings.append("น้ำหนักเป้าหมายทำให้ BMI ต่ำกว่า 18.5 ซึ่งต่ำกว่าเกณฑ์ปกติ ควรปรึกษาผู้เชี่ยวชาญ")

    notes = [
        "TDEE รวมการออกกำลังกายตามระดับกิจกรรมที่เลือกไว้แล้ว อย่านำแคลอรีจากหน้า Tracker มาบวกเพิ่มซ้ำ",
        "ระยะเวลาถึงเป้าหมายเป็นค่าประมาณคร่าวๆ เพราะ TDEE จะเปลี่ยนไปเมื่อน้ำหนักเปลี่ยน ควรปรับแผนทุก 2-4 สัปดาห์ตามผลจริง",
        "สัดส่วนสารอาหารตาม Somatotype เป็นแนวทางเชิงประมาณ หลักฐานทางวิทยาศาสตร์รองรับจำกัด ให้ยึดผลลัพธ์จริงของร่างกายเป็นหลัก",
        "เป็นคำแนะนำทั่วไป ไม่ใช่คำแนะนำทางการแพทย์ หากมีโรคประจำตัว ตั้งครรภ์ หรือมีประวัติการกินผิดปกติ ควรปรึกษาแพทย์/นักกำหนดอาหาร",
    ]

    return NutritionPlan(
        gender=gender, goal=goal, activity_level=activity_level, somatotype=somatotype,
        bmi=round(bmi, 1), bmr=round(bmr), tdee=round(tdee), target_calories=round(target),
        calorie_adjustment=round(adjustment), weekly_weight_change_kg=round(weekly_change, 2),
        weeks_to_goal=weeks, macros=calc_macros(target, weight_kg, goal, somatotype),
        warnings=warnings, notes=notes,
    )