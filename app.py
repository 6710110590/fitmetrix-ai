import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from features import calc_bmi
from src.ui import empty_html, hero_html, inject_css, note_html

st.set_page_config(page_title="FitMetrix AI", layout="wide")
inject_css()

st.markdown(
    hero_html(
        "FitMetrix AI",
        "วางแผนออกกำลังกายและโภชนาการ ติดตามแคลอรี และตรวจท่า Squat แบบเรียลไทม์ ด้วย Machine Learning",
        ["Scikit-learn", "MediaPipe", "Streamlit"],
    ),
    unsafe_allow_html=True,
)

# ---------------- ขั้นตอนการใช้งาน ----------------
# สไตล์ลิงก์ในการ์ดจำกัดเฉพาะพื้นที่หลัก (stMain / section.main) ไม่ให้ไปกระทบลิงก์ในเมนูซ้าย
st.markdown(
    """
    <style>
    .fm-step{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
    .fm-badge{flex:none;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;
              background:rgba(34,211,166,.15);border:1px solid rgba(34,211,166,.5);color:#7FF0D1;font-weight:700;font-size:.85rem;}
    .fm-step-title{font-weight:700;font-size:1rem;}
    .fm-desc{color:#9AA6BF;font-size:.85rem;line-height:1.45;min-height:4.4em;margin-bottom:10px;}
    [data-testid="stMain"] [data-testid="stPageLink-NavLink"],
    section.main [data-testid="stPageLink-NavLink"]{box-sizing:border-box;min-height:2.4rem;justify-content:center;
             border:1px solid #26324A;border-radius:10px;}
    [data-testid="stMain"] [data-testid="stPageLink-NavLink"]:hover,
    section.main [data-testid="stPageLink-NavLink"]:hover{border-color:#22D3A6;}
    </style>
    """,
    unsafe_allow_html=True,
)

HOME = Path(__file__).name   # ชื่อไฟล์หน้าแรก (app.py) ใช้ทำปุ่ม "คุณอยู่ที่นี่" ให้หน้าตาเหมือนปุ่มของการ์ดอื่น
STEPS = [
    ("1", "โปรไฟล์", "กรอกข้อมูลร่างกายด้านล่าง", HOME),
    ("2", "สแกนรูปร่าง", "ประเมิน Somatotype จากข้อมูลและภาพถ่าย", "pages/1_Profile_Scanner.py"),
    ("3", "วางแผน", "BMR/TDEE แคลอรี สารอาหาร และตารางฝึก", "pages/2_Plan.py"),
    ("4", "ติดตามแคลอรี", "ทำนายแคลอรีที่เผาผลาญด้วยโมเดล ML", "pages/3_Tracker.py"),
    ("5", "ตรวจท่า Squat", "นับครั้งและเตือนท่าผิดแบบเรียลไทม์", "pages/4_Squat.py"),
]
for col, (num, title, desc, page) in zip(st.columns(5, gap="medium"), STEPS):
    with col, st.container(border=True):
        st.markdown(
            f'<div class="fm-step"><span class="fm-badge">{num}</span><span class="fm-step-title">{title}</span></div>'
            f'<div class="fm-desc">{desc}</div>',
            unsafe_allow_html=True,
        )
        if page == HOME:
            st.page_link(HOME, label="คุณอยู่ที่นี่", disabled=True)   # ปุ่มแบบเดียวกับการ์ดอื่น แต่กดไม่ได้
        elif page and (ROOT / page).exists():
            st.page_link(page, label="เปิดหน้านี้")

# ---------------- โปรไฟล์ ----------------
saved = st.session_state.get("profile", {})
left, right = st.columns([1.2, 1], gap="large")

with left:
    with st.container(border=True):
        st.subheader("ข้อมูลโปรไฟล์ของคุณ")
        with st.form("profile_form"):
            gender = st.radio(
                "เพศ", ["male", "female"],
                index=0 if saved.get("gender", "male") == "male" else 1,
                format_func=lambda g: "ชาย" if g == "male" else "หญิง",
                horizontal=True,
            )
            c1, c2 = st.columns(2)
            age = c1.number_input("อายุ (ปี)", 10, 100, int(saved.get("age", 25)))
            height = c2.number_input("ส่วนสูง (ซม.)", 100.0, 250.0, float(saved.get("height_cm", 170.0)), step=0.5)
            c3, c4 = st.columns(2)
            weight = c3.number_input("น้ำหนักปัจจุบัน (กก.)", 30.0, 250.0, float(saved.get("weight_kg", 65.0)), step=0.5)
            target = c4.number_input("น้ำหนักเป้าหมาย (กก.)", 30.0, 250.0, float(saved.get("target_weight_kg", 65.0)), step=0.5)
            submitted = st.form_submit_button("บันทึกโปรไฟล์", type="primary")
        st.caption("แผนโภชนาการและ Scanner ให้คำแนะนำสำหรับผู้ใหญ่ (18 ปีขึ้นไป)")

if submitted:
    st.session_state["profile"] = {
        "gender": gender, "age": age, "height_cm": height,
        "weight_kg": weight, "target_weight_kg": target,
    }

with right:
    with st.container(border=True):
        st.subheader("สรุปโปรไฟล์")
        profile = st.session_state.get("profile")
        if profile:
            if submitted:
                st.success("บันทึกแล้ว เลือกหน้าถัดไปจากการ์ดด้านบน")
            m1, m2 = st.columns(2)
            m1.metric("BMI", f"{calc_bmi(profile['weight_kg'], profile['height_cm']):.1f}")
            m2.metric(
                "น้ำหนัก", f"{profile['weight_kg']:.1f} กก.",
                f"{profile['target_weight_kg'] - profile['weight_kg']:+.1f} กก. ถึงเป้าหมาย", delta_color="off",
            )
            st.caption(
                f"{'ชาย' if profile['gender'] == 'male' else 'หญิง'} · {profile['age']} ปี · "
                f"สูง {profile['height_cm']:.0f} ซม. | BMI ใช้ประกอบการคำนวณเท่านั้น ไม่ใช่การวินิจฉัย"
            )
        else:
            st.markdown(empty_html("ยังไม่มีโปรไฟล์ กรอกข้อมูลทางซ้ายแล้วกดบันทึก"), unsafe_allow_html=True)

st.markdown(
    note_html("ข้อมูลและผลลัพธ์ทั้งหมดเป็นค่าประมาณเพื่อใช้ทั่วไป ไม่ใช่คำแนะนำทางการแพทย์หรือโภชนาการเฉพาะบุคคล"),
    unsafe_allow_html=True,
)