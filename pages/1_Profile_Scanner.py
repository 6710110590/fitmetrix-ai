import sys
from pathlib import Path
from src.ui import inject_css


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.metabolic import ProfileError
from src.scanner import (
    MODEL_PATH, SOMATOTYPE_LABELS_TH, SOMATOTYPE_SHORT_TH, PoseDetector, ScannerSetupError,
    analyze_body, analyze_detection, download_model, draw_overlay, load_image,
)

st.set_page_config(page_title="Scanner | FitMetrix AI", layout="centered")
inject_css()

@st.cache_resource
def load_detector() -> PoseDetector:
    return PoseDetector()


st.title("AI Somatotype Scanner")
st.caption("ประเมินรูปร่างเบื้องต้น (Ectomorph / Mesomorph / Endomorph) จากข้อมูลร่างกายและสัดส่วนในภาพถ่าย")
st.info(
    "ผลเป็นการประมาณเชิงกฎ ไม่ใช่การวัดหรือการวินิจฉัยทางการแพทย์ "
    "ระบบวัดความกว้างไหล่ เอว และสะโพกจาก 'เงาร่างกาย' ในภาพ 2 มิติ แล้วรวมกับข้อมูลตัวเลขที่กรอก "
    "เกณฑ์ที่ใช้ยังไม่ได้สอบเทียบกับภาพที่ติดป้ายจริง แอปประมวลผลภาพในหน่วยความจำและไม่บันทึกภาพลงดิสก์"
)

saved = st.session_state.get("profile", {})

# ---------------- ข้อมูลตัวเลข ----------------
st.subheader("ข้อมูลร่างกาย")
c1, c2 = st.columns(2)
gender = c1.radio(
    "เพศ", ["male", "female"],
    index=0 if saved.get("gender", "male") == "male" else 1,
    format_func=lambda g: "ชาย" if g == "male" else "หญิง", horizontal=True,
)
age = c2.number_input("อายุ (ปี)", 18, 100, max(18, int(saved.get("age", 25))))
height = c1.number_input("ส่วนสูง (ซม.)", 100.0, 250.0, float(saved.get("height_cm", 170.0)), step=0.5)
weight = c2.number_input("น้ำหนัก (กก.)", 30.0, 250.0, float(saved.get("weight_kg", 65.0)), step=0.5)

with st.expander("วัดรอบตัวเพิ่ม (ไม่บังคับ แต่ช่วยให้ประเมินไขมันแม่นขึ้น)"):
    st.caption("ใช้สายวัดแบบนุ่ม เอว: ระดับสะดือ | คอ: ใต้ลูกกระเดือก | สะโพก: จุดที่กว้างสุด (ผู้หญิงต้องกรอกสะโพกด้วย) | 0 = ไม่กรอก")
    m1, m2, m3 = st.columns(3)
    waist = m1.number_input("รอบเอว (ซม.)", 0.0, 200.0, 0.0, step=0.5)
    neck = m2.number_input("รอบคอ (ซม.)", 0.0, 80.0, 0.0, step=0.5)
    hip = m3.number_input("รอบสะโพก (ซม.)", 0.0, 200.0, 0.0, step=0.5)
measurements = {"waist_cm": waist or None, "neck_cm": neck or None, "hip_cm": hip or None}

# ---------------- ภาพถ่าย ----------------
st.subheader("ภาพถ่าย")
mode = st.radio(
    "วิธีวิเคราะห์", ["photo", "numbers"], horizontal=True,
    format_func={"photo": "ใช้ภาพถ่าย + ข้อมูลตัวเลข", "numbers": "ใช้ข้อมูลตัวเลขอย่างเดียว"}.get,
)

detector, image_file = None, None
if mode == "photo":
    with st.expander("วิธีถ่ายให้แม่นขึ้น", expanded=True):
        st.markdown(
            "- ยืนตรง หันหน้าเข้ากล้อง **กางแขนออกจากลำตัวเล็กน้อย (ประมาณ 20-30 องศา)** ให้เห็นช่องว่างระหว่างแขนกับลำตัว\n"
            "- ให้เห็นเต็มตัวตั้งแต่ศีรษะถึงเท้า **เหลือที่ว่างเหนือศีรษะ (รวมผม) และใต้เท้า ไม่ให้ชิดขอบภาพ** วางกล้องระดับเอว ห่างประมาณ 2-3 เมตร\n"
            "- ปล่อยแขนลงข้างลำตัวห่างเล็กน้อย **ไม่ยกมือสูง ไม่ใช้โทรศัพท์หรือแขนบังลำตัว** ให้เห็นแขนทั้งสองข้างชัดเจน\n"
            "- ใส่ชุดกระชับตัว (ชุดหลวมทำให้เงากว้างเกินจริง) ฉากหลังเรียบ แสงสว่างเพียงพอ ไม่มีคนหรือสิ่งของอื่นติดเงา"
        )
    try:
        detector = load_detector()
    except ScannerSetupError as e:
        st.error(str(e))
        st.caption("ระหว่างนี้เลือก 'ใช้ข้อมูลตัวเลขอย่างเดียว' ได้")
        if not MODEL_PATH.exists() and st.button("ดาวน์โหลดโมเดลท่าทาง (ประมาณ 6 MB)"):
            try:
                with st.spinner("กำลังดาวน์โหลด..."):
                    download_model()
                st.rerun()
            except ScannerSetupError as e2:
                st.error(str(e2))

    if detector is not None:
        tab_cam, tab_up = st.tabs(["ถ่ายด้วยกล้อง", "อัปโหลดรูป"])
        cam = tab_cam.camera_input("ถ่ายภาพยืนตรง เห็นเต็มตัว")
        up = tab_up.file_uploader("เลือกรูป", type=["jpg", "jpeg", "png"])
        image_file = cam or up

ready = mode == "numbers" or image_file is not None
if st.button("วิเคราะห์", type="primary", disabled=not ready):
    try:
        if mode == "photo":
            img = load_image(image_file)
            detection = detector.detect(img)
            result = analyze_detection(detection, gender, age, height, weight, **measurements)
            overlay = draw_overlay(img, detection, result.silhouette) if detection else None
        else:
            result, overlay = analyze_body(gender, age, height, weight, **measurements), None
    except ProfileError as e:
        for msg in e.errors:
            st.error(msg)
    else:
        st.session_state["scan_result"] = {"result": result.to_dict(), "overlay": overlay}

# ---------------- ผลลัพธ์ ----------------
res = st.session_state.get("scan_result")
if res:
    r = res["result"]
    sil = r["silhouette"]
    st.divider()
    if res["overlay"] is not None:
        caption = ("ฟ้า = ไหล่ | เหลือง = เอว (แคบสุด) | เขียว = สะโพก (กว้างสุด) | ส้ม = บริเวณแขนที่ตัดออกจากการวัด "
                   "ตรวจดูว่าเส้นอยู่ตรงตำแหน่งจริงของร่างกายหรือไม่"
                   if sil.get("ok") else
                   "เส้นฟ้า/เขียว = ตำแหน่งข้อต่อไหล่/สะโพก (ไม่ใช่ความกว้างจริงของร่างกาย)")
        st.image(res["overlay"], caption=caption)
    for msg in r["photo_errors"]:
        st.error(msg)
    if r["photo_errors"]:
        st.info("ใช้ภาพนี้วัดสัดส่วนไม่ได้ ผลด้านล่างจึงมาจากข้อมูลตัวเลขอย่างเดียว ลองถ่ายใหม่ตามคำแนะนำด้านบน")

    st.subheader(r["somatotype_label"])
    if r["agreement"] == "conflict":
        st.caption(f"ตัวเลขชี้ไปทาง {SOMATOTYPE_SHORT_TH[r['numeric_somatotype']]} | "
                   f"ภาพชี้ไปทาง {SOMATOTYPE_SHORT_TH[r['visual_somatotype']]} | ดูเหตุผลด้านล่าง")
    k1, k2, k3 = st.columns(3)
    k1.metric("BMI", r["bmi"])
    k2.metric("ไขมันโดยประมาณ", f"{r['body_fat_pct']}%")
    k3.metric("เอว:สะโพก (จากภาพ)", sil["waist_hip_ratio"] if sil.get("ok") else "-")
    st.caption("ความมั่นใจ: " + {"medium": "ปานกลาง", "low": "ต่ำ"}[r["confidence"]]
               + (f" | รูปทรง: {r['shape_label']}" if r["shape_label"] else ""))
    with st.expander("เหตุผลที่จัดเป็นแบบนี้", expanded=(r["agreement"] == "conflict")):
        for line in r["reasons"]:
            st.markdown(f"- {line}")
    if sil.get("ok"):
        with st.expander("ค่าที่วัดจากเงาในภาพ"):
            show = lambda v: "-" if v is None else v  # noqa: E731  (ไหล่อาจวัดไม่ได้เมื่อแขนยกสูง)
            st.table({
                "ตำแหน่ง": ["ไหล่", "เอว (แคบสุด)", "สะโพก (กว้างสุด)"],
                "กว้าง (พิกเซล)": [show(sil["shoulder_width_px"]), sil["waist_width_px"], sil["hip_width_px"]],
                "สัดส่วนต่อส่วนสูงตัว": [show(sil["shoulder_ratio"]), sil["waist_ratio"], sil["hip_ratio"]],
            })
            st.caption("ใช้เปรียบเทียบระหว่างภาพได้ ค่าเหล่านี้ถูกกระทบจากชุดที่หลวม ท่ายืน และมุมกล้อง")
    for msg in r["warnings"]:
        st.warning(msg)

    keys = list(SOMATOTYPE_LABELS_TH)
    choice = st.selectbox("ไม่เห็นด้วยกับผลนี้? ปรับเองได้", keys, index=keys.index(r["somatotype"]),
                          format_func=SOMATOTYPE_LABELS_TH.get)
    if st.button("ใช้ผลนี้ในแผน"):
        st.session_state["scan"] = {
            "somatotype": choice,
            "source": "scanner" if choice == r["somatotype"] else "manual",
            "body_fat_pct": r["body_fat_pct"],
        }
        st.success("บันทึกแล้ว ไปที่หน้า Plan เพื่อสร้างแผน ระบบจะใช้รูปร่างนี้ให้อัตโนมัติ")