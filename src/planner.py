"""
planner.py - Phase 2: ตารางฝึก (Rule-based) ตาม สภาพแวดล้อม x เป้าหมาย x รูปร่างที่ต้องการ x Somatotype

ไม่ใช้ ML ตรงนี้ เป็นกฎและคลังท่าที่แก้ไขเพิ่มเติมได้ง่ายใน EXERCISE_BANK / SPLITS
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracker import calc_hr_max  # noqa: E402  (ใช้สูตร HRmax ตัวเดียวกับ Tracker)

ENVIRONMENTS = ("home", "gym", "outdoor")
ENVIRONMENT_LABELS_TH = {"home": "ที่บ้าน", "gym": "ฟิตเนส", "outdoor": "กลางแจ้ง / สวนสาธารณะ"}
FOCUSES = ("hourglass", "v_shape", "balanced")
FOCUS_LABELS_TH = {
    "hourglass": "เอวคอด ก้นเด้ง (Hourglass)",
    "v_shape": "ไหล่กว้าง เอวเล็ก (V-Shape)",
    "balanced": "สมส่วนทั่วไป",
}
LEVELS = ("beginner", "intermediate")

# ----------------------------- คลังท่าออกกำลังกาย -----------------------------
EXERCISE_BANK = {
    "home": {
        "glutes": ["Glute Bridge", "Hip Thrust (หลังพิงโซฟา)", "Donkey Kick", "Single-leg Glute Bridge"],
        "legs": ["Bodyweight Squat", "Reverse Lunge", "Bulgarian Split Squat (เท้าหลังบนเก้าอี้)", "Wall Sit"],
        "chest": ["Push-up", "Incline Push-up (มือบนโต๊ะ)", "Decline Push-up (เท้าบนเก้าอี้)"],
        "back": ["Backpack/Dumbbell Row", "Superman", "Reverse Snow Angel"],
        "shoulders": ["Pike Push-up", "Lateral Raise (ขวดน้ำ/ดัมเบล)", "Plank Shoulder Tap"],
        "arms": ["Close-grip Push-up", "Backpack/Dumbbell Curl", "Chair Dip"],
        "core": ["Plank", "Dead Bug", "Bicycle Crunch", "Mountain Climber"],
    },
    "gym": {
        "glutes": ["Barbell Hip Thrust", "Romanian Deadlift", "Cable Kickback", "Bulgarian Split Squat"],
        "legs": ["Back Squat", "Leg Press", "Walking Lunge", "Leg Curl"],
        "chest": ["Bench Press", "Incline Dumbbell Press", "Cable Fly"],
        "back": ["Lat Pulldown", "Seated Cable Row", "One-arm Dumbbell Row", "Face Pull"],
        "shoulders": ["Overhead Press", "Lateral Raise", "Rear Delt Fly"],
        "arms": ["EZ-bar Curl", "Triceps Pushdown", "Hammer Curl"],
        "core": ["Cable Crunch", "Hanging Knee Raise", "Plank", "Pallof Press"],
    },
    "outdoor": {
        "glutes": ["Incline/Hill Walk", "Step-up (ม้านั่งสวนสาธารณะ)", "Single-leg Glute Bridge", "Broad Jump"],
        "legs": ["Walking Lunge", "Bodyweight Squat", "Hill Sprint", "Step-up"],
        "chest": ["Push-up", "Decline Push-up (เท้าบนม้านั่ง)", "Diamond Push-up"],
        "back": ["Pull-up / Chin-up (บาร์สวนสาธารณะ)", "Inverted Row (บาร์ต่ำ)", "Superman"],
        "shoulders": ["Pike Push-up", "Bear Crawl", "Plank to Down Dog"],
        "arms": ["Chin-up", "Bench Dip", "Diamond Push-up"],
        "core": ["Hanging Knee Raise", "Plank", "Mountain Climber", "Leg Raise"],
    },
}

# ----------------------------- การแบ่งวันฝึก ---------------------------------
# SPLITS[focus][จำนวนวัน] = [(ชื่อวัน, [กลุ่มกล้ามเนื้อ]), ...]
SPLITS = {
    "hourglass": {
        3: [("Glutes & Hamstrings", ["glutes", "legs"]),
            ("Upper Body Tone (หลัง/ไหล่/แขน)", ["back", "shoulders", "arms"]),
            ("Legs, Glutes & Core", ["legs", "glutes", "core"])],
        4: [("Glutes & Legs A", ["glutes", "legs"]),
            ("Upper Body Tone", ["back", "shoulders", "chest"]),
            ("Glutes & Legs B", ["glutes", "legs"]),
            ("Core & Arms", ["core", "arms"])],
        5: [("Glutes Focus", ["glutes", "legs"]),
            ("Back & Shoulders", ["back", "shoulders"]),
            ("Legs & Glutes", ["legs", "glutes"]),
            ("Core & Arms", ["core", "arms"]),
            ("Full Body Light", ["glutes", "chest", "back", "core"])],
    },
    "v_shape": {
        3: [("Back & Shoulders", ["back", "shoulders"]),
            ("Chest & Arms", ["chest", "arms"]),
            ("Legs & Core", ["legs", "glutes", "core"])],
        4: [("Back & Biceps", ["back", "arms"]),
            ("Chest & Shoulders", ["chest", "shoulders"]),
            ("Legs & Core", ["legs", "glutes", "core"]),
            ("Shoulders & Back", ["shoulders", "back"])],
        5: [("Back & Biceps", ["back", "arms"]),
            ("Chest & Triceps", ["chest", "arms"]),
            ("Legs", ["legs", "glutes"]),
            ("Shoulders & Core", ["shoulders", "core"]),
            ("Back & Chest", ["back", "chest"])],
    },
    "balanced": {
        3: [("Full Body A", ["legs", "chest", "back", "core"]),
            ("Full Body B", ["glutes", "shoulders", "back", "core"]),
            ("Full Body C", ["legs", "chest", "arms", "core"])],
        4: [("Upper A", ["chest", "back", "shoulders"]),
            ("Lower A", ["legs", "glutes", "core"]),
            ("Upper B", ["back", "chest", "arms"]),
            ("Lower B", ["glutes", "legs", "core"])],
        5: [("Upper A", ["chest", "back"]),
            ("Lower A", ["legs", "glutes"]),
            ("Shoulders & Arms", ["shoulders", "arms", "core"]),
            ("Lower B", ["glutes", "legs"]),
            ("Upper B", ["back", "chest", "core"])],
    },
}

SET_REP_SCHEMES = {          # (เซ็ต, จำนวนครั้ง)
    "build_muscle": ("3-4", "8-12"),
    "lose_fat": ("3", "12-15"),
    "maintain": ("3", "10-12"),
}
BASE_CARDIO_SESSIONS = {"lose_fat": 3, "build_muscle": 1, "maintain": 2}


def default_focus(gender: str) -> str:
    return "hourglass" if str(gender).lower() == "female" else "v_shape"


def _cardio_plan(goal: str, somatotype: str, age: float, level: str) -> dict:
    sessions = BASE_CARDIO_SESSIONS[goal]
    if somatotype == "endomorph":
        sessions += 1
    elif somatotype == "ectomorph":
        sessions = max(1, sessions - 1)
    sessions = min(sessions, 3 if level == "beginner" else 4)   # มือใหม่ไม่เกิน 3 ครั้ง/สัปดาห์

    hr_max = calc_hr_max(age)
    z2 = (round(hr_max * 0.60), round(hr_max * 0.70))
    minutes = "30-40" if goal == "lose_fat" else "20-30"
    text = f"คาร์ดิโอเบา-ปานกลาง {minutes} นาที ใน Zone 2 (ชีพจรประมาณ {z2[0]}-{z2[1]} bpm) เช่น เดินเร็ว วิ่งเหยาะ ปั่นจักรยาน"
    if goal == "lose_fat" and sessions >= 3:
        text += " และเปลี่ยน 1 ครั้งเป็นแบบ Interval (วิ่งเร็ว 30 วินาที สลับเดินพัก 90 วินาที x 8-10 รอบ) ถ้าร่างกายพร้อม"
    return {"sessions_per_week": sessions, "zone2_hr_range": z2, "description": text}


def build_workout_plan(
    gender: str,
    age: float,
    goal: str,
    environment: str,
    days_per_week: int = 4,
    focus: str | None = None,
    somatotype: str = "mesomorph",
    level: str = "beginner",
) -> dict:
    if environment not in ENVIRONMENTS:
        raise ValueError(f"environment ต้องเป็นหนึ่งใน {ENVIRONMENTS}")
    if goal not in SET_REP_SCHEMES:
        raise ValueError(f"goal ต้องเป็นหนึ่งใน {tuple(SET_REP_SCHEMES)}")
    focus = focus or default_focus(gender)
    if focus not in FOCUSES:
        raise ValueError(f"focus ต้องเป็นหนึ่งใน {FOCUSES}")
    if level not in LEVELS:
        raise ValueError(f"level ต้องเป็นหนึ่งใน {LEVELS}")

    days = min(max(int(days_per_week), 3), 5)
    sets, reps = SET_REP_SCHEMES[goal]
    if level == "beginner":
        sets = "2-3"

    bank = EXERCISE_BANK[environment]
    sessions = []
    for i, (title, groups) in enumerate(SPLITS[focus][days], start=1):
        per_group = 3 if len(groups) <= 2 else 2
        exercises = [
            {"name": name, "muscle": group, "sets": sets, "reps": reps}
            for group in groups
            for name in bank[group][:per_group]
        ]
        sessions.append({"day": f"Day {i}", "title": title, "exercises": exercises})

    cardio = _cardio_plan(goal, somatotype, age, level)
    notes = [
        "วอร์มอัพ 5-10 นาทีก่อนเริ่ม และยืดเหยียดหลังฝึก",
        "เพิ่มความหนัก (น้ำหนัก/จำนวนครั้ง/ความยากของท่า) ทีละน้อยเมื่อทำครบเซ็ตได้สบาย และเว้นกลุ่มกล้ามเนื้อเดิมอย่างน้อย 48 ชั่วโมง",
        "ไม่มีท่าที่ลดไขมันเฉพาะจุดได้ ไขมันจะลดตามภาพรวมจากแคลอรีที่เหมาะสม ท่าที่เน้นช่วยสร้างรูปร่างของกล้ามเนื้อเท่านั้น",
        "หยุดพักทันทีหากรู้สึกเจ็บแปลบ เวียนหัว หรือหายใจไม่ทัน เป็นคำแนะนำทั่วไป ไม่ใช่คำแนะนำทางการแพทย์",
    ]
    if somatotype == "ectomorph":
        notes.append("Ectomorph: เน้นท่าหลายข้อต่อ พักระหว่างเซ็ตให้พอ และจำกัดคาร์ดิโอเพื่อไม่ให้แคลอรีขาด (แนวทางเชิงประมาณ)")
    elif somatotype == "endomorph":
        notes.append("Endomorph: เพิ่มคาร์ดิโอและลดเวลาพักระหว่างเซ็ตเล็กน้อยได้ (แนวทางเชิงประมาณ)")

    return {
        "goal": goal, "environment": environment, "focus": focus, "level": level,
        "days_per_week": days, "sessions": sessions, "cardio": cardio, "notes": notes,
    }