import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.squat import ISSUE_TEXT, SquatConfig
from src.ui import (
    DEMO_CAPTIONS, empty_html, header_html, inject_css, note_html, rep_chart_html, rep_stats,
    show_html, squat_demo_html,
)

st.set_page_config(page_title="Squat | FitMetrix AI", layout="wide")
inject_css()
st.markdown(
    header_html("นับ Squat และเตือนท่าผิดแบบเรียลไทม์",
                "ใช้เว็บแคมของเครื่องนี้จับจุดข้อต่อ นับจำนวนครั้ง และเตือนเมื่อท่าไม่ถูกต้อง"),
    unsafe_allow_html=True,
)
st.markdown(
    note_html("คำแนะนำเบื้องต้นจากการประมาณมุมข้อต่อ ไม่ใช่คำแนะนำทางการแพทย์หรือกายภาพบำบัด "
              "ระบบวัด 'หลังงอ' ไม่ได้ วัดได้แค่ลำตัวเอียงจากแนวดิ่ง เกณฑ์ที่ใช้ยังไม่ได้สอบเทียบ "
              "ประมวลผลบนเครื่องและไม่บันทึกภาพหรือวิดีโอ"),
    unsafe_allow_html=True,
)
st.warning("ฟีเจอร์นี้เปิดหน้าต่างกล้องบนเครื่องที่รันแอป ใช้ได้เฉพาะเมื่อรันแอปบนเครื่องที่ต่อกล้อง (localhost) ไม่ทำงานเมื่อ deploy ออนไลน์")

left, right = st.columns([1.15, 1], gap="large")

# ---------------- ควบคุมกล้อง ----------------
with left:
    with st.container(border=True):
        st.subheader("เริ่มนับ")
        view = st.radio(
            "มุมกล้อง", ["front", "side"], horizontal=True,
            format_func={"front": "ด้านหน้า (กล้องโน้ตบุ๊กอยู่ตรงหน้า)", "side": "ด้านข้าง (ยืนหันข้างให้กล้อง)"}.get,
        )
        c1, c2 = st.columns(2)
        camera = c1.number_input("หมายเลขกล้อง", 0, 5, 0)
        sound = c2.checkbox("เสียงเตือน (Windows)", value=True)

        proc = st.session_state.get("squat_proc")
        running = proc is not None and proc.poll() is None
        b1, b2 = st.columns(2)
        if b1.button("เปิดกล้องและเริ่มนับ", type="primary", disabled=running, use_container_width=True):
            cmd = [sys.executable, str(ROOT / "live_squat.py"), "--view", view, "--camera", str(int(camera))]
            if not sound:
                cmd.append("--no-sound")
            st.session_state["squat_proc"] = subprocess.Popen(cmd, cwd=str(ROOT))
            st.rerun()
        if b2.button("หยุด", disabled=not running, use_container_width=True):
            proc.terminate()
            st.rerun()
        if running:
            st.success("กำลังทำงาน ดูที่หน้าต่างกล้อง (อาจอยู่ด้านหลังหน้าต่างนี้) กด Q ในหน้าต่างนั้นเมื่อเสร็จ แล้วกลับมาที่หน้านี้เพื่อดูสรุป")
            st.button("รีเฟรชสถานะ")
        st.caption("กรอบรอบภาพในหน้าต่างกล้อง: สีเขียว = ท่าปกติ/ท่าดี · สีเหลือง = ยังไม่พร้อมหรือมีข้อเตือน · สีแดง = ผิดท่าหรือครั้งที่ไม่นับ")

    with st.expander("วิธีตั้งกล้องและยืน", expanded=False):
        st.markdown(
            "- ตั้งโน้ตบุ๊กบนโต๊ะหรือเก้าอี้ให้กล้องอยู่ระดับเอวถึงอก ยืนห่างประมาณ **2-3 เมตร** ให้เห็นตั้งแต่ศีรษะถึงเท้า\n"
            "- **ด้านหน้า:** หันหน้าเข้ากล้อง ยืนนิ่งๆ ประมาณ 1 วินาทีให้ระบบตั้งค่าเริ่มต้น (ขึ้น Calibrating) "
            "เตือนได้: ลงไม่ลึก เข่าหุบ สะโพกเอียง\n"
            "- **ด้านข้าง:** ยืนหันข้างให้กล้อง เตือนได้: ลงไม่ลึก ลำตัวโน้มไปข้างหน้ามากไป\n"
            "- ใส่ชุดที่เห็นรูปร่างขาชัด ฉากหลังไม่รก แสงสว่างพอ\n"
            "- ปุ่มในหน้าต่างกล้อง: **Q** จบ | **R** เริ่มนับใหม่ | **C** ตั้งค่าเริ่มต้นใหม่"
        )

# ---------------- ภาพคนเส้นสาธิตท่า ----------------
with right:
    with st.container(border=True):
        st.markdown("**ท่าตัวอย่าง**" + ("  (มุมด้านข้าง)" if view == "side" else "  (มุมด้านหน้า)"))
        show_html(squat_demo_html("side" if view == "side" else "front"), height=290)
        st.caption(DEMO_CAPTIONS["side" if view == "side" else "front"])
        st.caption("ภาพประกอบวาดจากโค้ด ใช้อธิบายท่าเท่านั้น ไม่ได้มาจากการตรวจจับจริง")


# ---------------- สรุปเซสชัน ----------------
def log_label(path: Path) -> str:
    try:
        return datetime.strptime(path.stem[len("squat_"):], "%Y%m%d_%H%M%S").strftime("%d/%m/%Y %H:%M:%S")
    except ValueError:
        return path.name


def load_cfg(data: dict) -> SquatConfig:
    """อ่านเกณฑ์ที่ใช้ตอนบันทึก (ไม่มีหรืออ่านไม่ได้ ใช้ค่าตั้งต้นของมุมกล้องนั้น)"""
    try:
        return SquatConfig(**data["config"])
    except Exception:  # noqa: BLE001
        return SquatConfig(view=data.get("view", "front"))


def issue_th(code: str) -> str:
    return ISSUE_TEXT.get(code, {"th": code})["th"]


st.divider()
logs = sorted((ROOT / "logs").glob("squat_*.json")) if (ROOT / "logs").exists() else []
st.subheader("สรุปเซสชัน")
if not logs:
    st.markdown(empty_html("ยังไม่มีข้อมูล กด 'เปิดกล้องและเริ่มนับ' แล้วออกกำลังกาย จากนั้นกด Q เพื่อจบ สรุปจะแสดงที่นี่"),
                unsafe_allow_html=True)
else:
    chosen = logs[-1]
    if len(logs) > 1:
        chosen = st.selectbox("เลือกเซสชัน", list(reversed(logs)), format_func=log_label)
    data = json.loads(chosen.read_text(encoding="utf-8"))
    cfg = load_cfg(data)
    _up, _start, count_thr, good_thr = cfg.thresholds()
    stats = rep_stats(data["events"], count_thr, good_thr)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("นับได้", f"{data['reps']} ครั้ง", f"พยายามทั้งหมด {data['attempts']}", delta_color="off")
    m2.metric("ครั้งที่ไม่มีข้อเตือน", f"{stats['good_reps']} ครั้ง")
    m3.metric("ลึกเฉลี่ย", f"{stats['avg_depth_pct']:.0f}%" if stats["avg_depth_pct"] is not None else "-")
    m4.metric("เวลาต่อครั้งเฉลี่ย", f"{stats['avg_duration']:.1f} วินาที" if stats["avg_duration"] is not None else "-")
    st.caption(f"มุมกล้อง: {'ด้านหน้า' if data['view'] == 'front' else 'ด้านข้าง'} | ไฟล์: logs/{chosen.name} | "
               "เก็บเฉพาะตัวเลขสรุป ไม่มีภาพหรือวิดีโอ")

    with st.container(border=True):
        st.markdown("**ความลึกของแต่ละครั้ง**")
        show_html(rep_chart_html(data["events"], count_thr, good_thr, {k: v["th"] for k, v in ISSUE_TEXT.items()}),
                  height=300)
        st.caption("ความสูงแท่งคือความลึกเทียบกับเกณฑ์ตั้งต้นที่ยังไม่ได้สอบเทียบ (100% = ลึกถึงเกณฑ์ดี) "
                   "ไม่ใช่คะแนนคุณภาพท่าที่แท้จริง เอาเมาส์ชี้แท่งเพื่อดูรายละเอียด")

    if data["issue_counts"]:
        with st.container(border=True):
            st.markdown("**ข้อเตือนที่เจอ**")
            for code, n in data["issue_counts"].items():
                st.markdown(f"- {issue_th(code)}: {n} ครั้ง")
    else:
        st.success("ไม่พบข้อเตือนตลอดเซสชัน")

    with st.expander("รายละเอียดทีละครั้ง"):
        st.table([
            {"ครั้งที่": e["index"] if e["counted"] else "-", "นับ": "นับ" if e["counted"] else "ไม่นับ",
             "ค่าต่ำสุด": e["min_signal"], "เวลา (วินาที)": e["duration"],
             "ข้อเตือน": ", ".join(issue_th(c) for c in e["issues"]) or "-"}
            for e in data["events"]
        ])