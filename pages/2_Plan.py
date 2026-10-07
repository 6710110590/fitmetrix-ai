import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.metabolic import (
    ACTIVITY_LABELS_TH, GOAL_LABELS_TH, GOALS, ProfileError, build_nutrition_plan,
)
from src.planner import (
    ENVIRONMENT_LABELS_TH, ENVIRONMENTS, FOCUS_LABELS_TH, FOCUSES,
    build_workout_plan, default_focus,
)
from src.ui import (
    PALETTE, donut_html, empty_html, header_html, inject_css, note_html,
    projection_html, projection_points, show_html,
)

st.set_page_config(page_title="Plan | FitMetrix AI", layout="wide")
inject_css()
st.markdown(
    header_html("แผนออกกำลังกายและโภชนาการ", "คำนวณ BMR/TDEE เป้าหมายแคลอรี สัดส่วนสารอาหาร และตารางฝึกตามเป้าหมายของคุณ"),
    unsafe_allow_html=True,
)

saved = st.session_state.get("profile", {})
scan = st.session_state.get("scan", {})

SOMATOTYPE_LABELS = {
    "unknown": "ยังไม่ทราบ (ใช้แบบสมดุล)",
    "ectomorph": "Ectomorph (ผอม เผาผลาญไว สร้างกล้ามยาก)",
    "mesomorph": "Mesomorph (สมส่วน สร้างกล้ามง่าย)",
    "endomorph": "Endomorph (เก็บไขมันง่าย โครงใหญ่)",
}
INTENSITY_LABELS = {"gentle": "ค่อยเป็นค่อยไป", "standard": "มาตรฐาน (แนะนำ)", "aggressive": "เข้มข้น"}
INTENSITY_PCT = {  # (ลดไขมัน, สร้างกล้าม)
    "gentle": (0.10, 0.05), "standard": (None, None), "aggressive": (0.20, 0.15),
}

left, right = st.columns([1, 1.9], gap="large")

# ---------------- ฟอร์มตั้งค่าแผน ----------------
with left:
    with st.container(border=True):
        st.subheader("ตั้งค่าแผน")
        if not saved:
            st.caption("ยังไม่มีโปรไฟล์ที่บันทึกไว้ กรอกด้านล่างได้เลย หรือไปหน้าหลักเพื่อบันทึกโปรไฟล์")
        with st.form("plan_form"):
            gender = st.radio(
                "เพศ", ["male", "female"],
                index=0 if saved.get("gender", "male") == "male" else 1,
                format_func=lambda g: "ชาย" if g == "male" else "หญิง", horizontal=True,
            )
            c1, c2 = st.columns(2)
            age = c1.number_input("อายุ (ปี)", 10, 100, int(saved.get("age", 25)))
            height = c2.number_input("ส่วนสูง (ซม.)", 100.0, 250.0, float(saved.get("height_cm", 170.0)), step=0.5)
            c3, c4 = st.columns(2)
            weight = c3.number_input("น้ำหนักปัจจุบัน (กก.)", 30.0, 250.0, float(saved.get("weight_kg", 65.0)), step=0.5)
            target_weight = c4.number_input("น้ำหนักเป้าหมาย (กก.)", 30.0, 250.0, float(saved.get("target_weight_kg", 65.0)), step=0.5)

            goal = st.radio("เป้าหมาย", list(GOALS), format_func=GOAL_LABELS_TH.get, horizontal=True)
            intensity = st.select_slider("ความเข้มข้นของแผน", list(INTENSITY_LABELS), value="standard", format_func=INTENSITY_LABELS.get)
            activity_level = st.selectbox("ระดับกิจกรรมในชีวิตประจำวัน", list(ACTIVITY_LABELS_TH), index=1, format_func=ACTIVITY_LABELS_TH.get)
            soma_keys = list(SOMATOTYPE_LABELS)
            somatotype = st.selectbox(
                "รูปร่าง (Somatotype)", soma_keys,
                index=soma_keys.index(scan["somatotype"]) if scan.get("somatotype") in soma_keys else 0,
                format_func=SOMATOTYPE_LABELS.get,
            )
            st.caption("ใช้ผลจากหน้า Scanner ให้อัตโนมัติ" if scan else "ไปหน้า Scanner เพื่อประเมินรูปร่าง หรือเลือกเอง/เลือก 'ยังไม่ทราบ' ได้")

            environment = st.selectbox("สถานที่ฝึก", list(ENVIRONMENTS), format_func=ENVIRONMENT_LABELS_TH.get)
            days = st.slider("วันฝึกต่อสัปดาห์", 3, 5, 4)
            focus = st.selectbox("รูปร่างที่ต้องการ", list(FOCUSES), index=list(FOCUSES).index(default_focus(gender)),
                                 format_func=FOCUS_LABELS_TH.get)
            level = st.radio("ระดับ", ["beginner", "intermediate"],
                             format_func={"beginner": "มือใหม่", "intermediate": "ฝึกมาแล้ว"}.get, horizontal=True)
            submitted = st.form_submit_button("สร้างแผน", type="primary")

if submitted:
    pct_lose, pct_gain = INTENSITY_PCT[intensity]
    adjust_pct = pct_lose if goal == "lose_fat" else pct_gain if goal == "build_muscle" else None
    soma = "mesomorph" if somatotype == "unknown" else somatotype
    try:
        nutrition = build_nutrition_plan(
            gender, age, height, weight, target_weight, activity_level, goal, soma, adjust_pct,
        )
    except ProfileError as e:
        with left:
            for msg in e.errors:
                st.error(msg)
    else:
        workout = build_workout_plan(gender, age, goal, environment, days, focus, soma, level)
        st.session_state["plan"] = {
            "nutrition": nutrition.to_dict(), "workout": workout,
            "inputs": {"weight": weight, "target_weight": target_weight},
        }


# ---------------- ผลลัพธ์ ----------------
def nutrition_tab(n: dict, inputs: dict) -> None:
    m1, m2, m3 = st.columns(3)
    m1.metric("BMR", f"{n['bmr']:,} kcal")
    m2.metric("TDEE", f"{n['tdee']:,} kcal")
    m3.metric("เป้าหมาย/วัน", f"{n['target_calories']:,} kcal", f"{n['calorie_adjustment']:+,} kcal", delta_color="off")
    st.caption(f"BMI {n['bmi']} | น้ำหนักเปลี่ยนโดยประมาณ {n['weekly_weight_change_kg']:+.2f} กก./สัปดาห์"
               + (f" | ถึงเป้าหมายราว {n['weeks_to_goal']} สัปดาห์" if n["weeks_to_goal"] else ""))
    for msg in n["warnings"]:
        st.warning(msg)

    macros = n["macros"]
    colors = {"protein": PALETTE["protein"], "carbs": PALETTE["carbs"], "fat": PALETTE["fat"]}
    names = {"protein": "โปรตีน", "carbs": "คาร์โบไฮเดรต", "fat": "ไขมัน"}
    segments = [
        (names[k], macros[k]["grams"], colors[k], f"{macros[k]['grams']} กรัม · {macros[k]['pct']}% · {macros[k]['kcal']} kcal")
        for k in ("protein", "carbs", "fat")
    ]
    with st.container(border=True):
        st.markdown("**สัดส่วนสารอาหารต่อวัน**")
        show_html(donut_html(segments, f"{n['target_calories']:,}", "kcal/วัน"), height=250)
        st.caption(f"โปรตีนประมาณ {macros['protein_g_per_kg']} กรัม/กก.น้ำหนักตัว")

    with st.container(border=True):
        st.markdown("**ประมาณการน้ำหนักตามแผน**")
        points = projection_points(inputs["weight"], inputs["target_weight"], n["weekly_weight_change_kg"])
        if points:
            show_html(projection_html(points, inputs["target_weight"], width=640), height=290)
            st.caption("เป็นค่าประมาณที่สมมติอัตราการเปลี่ยนคงที่ ไม่ได้คิดว่า TDEE เปลี่ยนตามน้ำหนัก ควรปรับแผนทุก 2-4 สัปดาห์ตามผลจริง")
        else:
            st.caption("ไม่แสดงกราฟ: เป้าหมายคือคงรูปร่าง น้ำหนักไม่เปลี่ยน หรือน้ำหนักเป้าหมายสวนทางกับเป้าหมายที่เลือก")


def workout_tab(w: dict) -> None:
    st.caption(f"{ENVIRONMENT_LABELS_TH[w['environment']]} | {FOCUS_LABELS_TH[w['focus']]} | {w['days_per_week']} วัน/สัปดาห์")
    for s in w["sessions"]:
        with st.expander(f"{s['day']}: {s['title']}", expanded=(s["day"] == "Day 1")):
            st.dataframe(
                pd.DataFrame(s["exercises"]).rename(columns={"name": "ท่า", "muscle": "กลุ่มกล้ามเนื้อ", "sets": "เซ็ต", "reps": "จำนวนครั้ง"}),
                hide_index=True, use_container_width=True,
            )
    with st.container(border=True):
        c1, c2 = st.columns([1, 2.5])
        c1.metric("คาร์ดิโอ", f"{w['cardio']['sessions_per_week']} ครั้ง/สัปดาห์")
        c2.markdown(w["cardio"]["description"])


with right:
    plan = st.session_state.get("plan")
    if not plan:
        st.markdown(empty_html("กรอกข้อมูลทางซ้ายแล้วกด 'สร้างแผน' ผลจะแสดงที่นี่"), unsafe_allow_html=True)
    else:
        n, w = plan["nutrition"], plan["workout"]
        tab_nutrition, tab_workout, tab_notes = st.tabs(["โภชนาการ", "ตารางฝึก", "หมายเหตุ"])
        with tab_nutrition:
            nutrition_tab(n, plan["inputs"])
        with tab_workout:
            workout_tab(w)
        with tab_notes:
            for line in n["notes"] + w["notes"]:
                st.markdown(f"- {line}")
            st.markdown(note_html("เป็นคำแนะนำทั่วไป ไม่ใช่คำแนะนำทางการแพทย์"), unsafe_allow_html=True)