import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.tracker import CalorieTracker, UserProfile, ValidationError, Workout, zone_table
from src.ui import empty_html, header_html, inject_css, note_html, show_html, zone_bar_html

st.set_page_config(page_title="Tracker | FitMetrix AI", layout="wide")
inject_css()


@st.cache_resource
def load_tracker() -> CalorieTracker:
    return CalorieTracker(policy="fallback")


tracker = load_tracker()
for w in tracker.load_warnings:
    st.warning(w)

st.markdown(
    header_html("Real-time Tracker", "ทำนายแคลอรีที่เผาผลาญจากเวลาและชีพจร พร้อมจำแนก Heart Rate Zone"),
    unsafe_allow_html=True,
)

saved = st.session_state.get("profile", {})
left, right = st.columns([1, 1.9], gap="large")

# ---------------- ฟอร์มกรอกข้อมูล ----------------
with left:
    with st.container(border=True):
        st.subheader("บันทึกกิจกรรม")
        if not saved:
            st.caption("ยังไม่มีโปรไฟล์ที่บันทึกไว้ กรอกด้านล่างได้เลย หรือไปหน้าหลักเพื่อบันทึกโปรไฟล์")
        with st.form("tracker_form"):
            gender = st.radio(
                "เพศ", ["male", "female"],
                index=0 if saved.get("gender", "male") == "male" else 1,
                format_func=lambda g: "ชาย" if g == "male" else "หญิง", horizontal=True,
            )
            c1, c2 = st.columns(2)
            age = c1.number_input("อายุ (ปี)", 10, 100, int(saved.get("age", 25)))
            height = c2.number_input("ส่วนสูง (ซม.)", 100.0, 250.0, float(saved.get("height_cm", 170.0)), step=0.5)
            weight = st.number_input("น้ำหนัก (กก.)", 30.0, 250.0, float(saved.get("weight_kg", 65.0)), step=0.5)
            activity = st.radio(
                "ประเภทกิจกรรม", ["cardio", "strength", "other"],
                format_func={"cardio": "คาร์ดิโอ (วิ่ง/ปั่น/ว่ายน้ำ)", "strength": "เวท / Strength", "other": "อื่นๆ"}.get,
                horizontal=True,
            )
            d1, d2 = st.columns(2)
            duration = d1.number_input("เวลา (นาที)", 1.0, 600.0, 20.0, step=1.0)
            heart_rate = d2.number_input("ชีพจรเฉลี่ย (ครั้ง/นาที)", 40, 230, 110)
            submitted = st.form_submit_button("คำนวณแคลอรี", type="primary")

# ---------------- คำนวณ ----------------
if submitted:
    try:
        result = tracker.predict(
            UserProfile(gender, age, height, weight),
            Workout(duration_min=duration, heart_rate=heart_rate, activity=activity),
        )
    except ValidationError as e:
        with left:
            for msg in e.errors:
                st.error(msg)
    else:
        st.session_state.setdefault("workout_log", []).append({
            "กิจกรรม": activity, "เวลา (นาที)": duration, "ชีพจร": heart_rate,
            "แคลอรี (kcal)": result.calories, "Zone": result.zone["name"],
            "วิธีคำนวณ": "โมเดล ML" if result.method == "ml_model" else "สูตร Keytel",
        })
        st.session_state["last_track"] = {"result": result.to_dict(), "heart_rate": heart_rate, "age": age}

# ---------------- ผลลัพธ์ ----------------
with right:
    last = st.session_state.get("last_track")
    if not last:
        st.markdown(empty_html("กรอกข้อมูลทางซ้ายแล้วกด 'คำนวณแคลอรี' ผลจะแสดงที่นี่"), unsafe_allow_html=True)
    else:
        r = last["result"]
        m1, m2, m3 = st.columns(3)
        m1.metric("แคลอรีที่เผาผลาญ", f"{r['calories']:.0f} kcal")
        m2.metric("Heart Rate Zone", f"Zone {r['zone']['zone']}" if r["zone"]["zone"] else "ต่ำกว่า Zone 1",
                  f"{r['zone']['pct_hrmax']}% ของ HRmax", delta_color="off")
        m3.metric("HRmax โดยประมาณ", f"{r['hr_max']:.0f} bpm")

        label = {"high": "ความมั่นใจสูง (โมเดล ML)", "medium": "ค่าประมาณจากสูตรมาตรฐาน", "low": "ความมั่นใจต่ำ"}
        st.caption(f"{r['zone']['name']} · {r['zone']['description']} | {label[r['confidence']]}")
        for msg in r["warnings"]:
            st.warning(msg)

        with st.container(border=True):
            st.markdown("**ชีพจรของคุณเทียบกับแต่ละโซน**")
            show_html(zone_bar_html(last["heart_rate"], r["hr_max"], r["zone"]["zone"]), height=150)

        with st.expander("ตารางช่วงชีพจรของแต่ละโซน (ตามอายุที่กรอก)"):
            st.table(pd.DataFrame(zone_table(last["age"])))

    log = st.session_state.get("workout_log", [])
    if log:
        with st.container(border=True):
            st.markdown("**บันทึกการออกกำลังกายในเซสชันนี้**")
            log_df = pd.DataFrame(log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            t1, t2 = st.columns([1, 3])
            t1.metric("รวมทั้งหมด", f"{log_df['แคลอรี (kcal)'].sum():.0f} kcal")
            if t2.button("ล้างบันทึก"):
                st.session_state["workout_log"] = []
                st.session_state.pop("last_track", None)
                st.rerun()

    st.markdown(note_html("ผลลัพธ์เป็นค่าประมาณ ไม่ใช่คำแนะนำทางการแพทย์"), unsafe_allow_html=True)