from pathlib import Path

import pytest

from tests.fake_streamlit import run_page

ROOT = Path(__file__).resolve().parent.parent
APP, PLAN, TRACKER = ROOT / "app.py", ROOT / "pages" / "2_Plan.py", ROOT / "pages" / "3_Tracker.py"

PROFILE = {"gender": "male", "age": 30, "height_cm": 180.0, "weight_kg": 80.0, "target_weight_kg": 75.0}


def metrics(st):
    return {c[1][0]: c[1][1] for c in st.called("metric")}


# ---------------- หน้าแรก ----------------
def test_home_renders_without_profile():
    st = run_page(APP)
    assert st.called("set_page_config")[0][2]["layout"] == "wide"
    assert any("FitMetrix AI" in t for t in st.texts("markdown"))
    assert len(st.called("page_link")) == 4                       # 4 หน้าที่มีไฟล์อยู่จริง
    assert not st.called("metric") and any("ยังไม่มีโปรไฟล์" in t for t in st.texts("markdown"))


def test_home_saves_profile_on_submit_and_shows_summary():
    st = run_page(APP, submit=True, values={"อายุ (ปี)": 30, "ส่วนสูง (ซม.)": 180.0, "น้ำหนักปัจจุบัน (กก.)": 80.0,
                                              "น้ำหนักเป้าหมาย (กก.)": 75.0})
    p = st.session_state["profile"]
    assert (p["gender"], p["age"], p["height_cm"], p["weight_kg"], p["target_weight_kg"]) == ("male", 30, 180.0, 80.0, 75.0)
    m = metrics(st)
    assert m["BMI"] == "24.7" and m["น้ำหนัก"] == "80.0 กก."
    assert any("-5.0 กก. ถึงเป้าหมาย" in str(c[1][2]) for c in st.called("metric"))
    assert st.called("success")


def test_home_shows_saved_profile_without_resubmit():
    st = run_page(APP, session={"profile": PROFILE})
    assert metrics(st)["BMI"] == "24.7" and not st.called("success")


# ---------------- หน้า Plan ----------------
def test_plan_empty_state_before_submit():
    st = run_page(PLAN)
    assert any("ผลจะแสดงที่นี่" in t for t in st.texts("markdown")) and not st.html


def test_plan_builds_full_result_with_charts():
    st = run_page(PLAN, submit=True, session={"profile": PROFILE})
    plan = st.session_state["plan"]
    assert plan["nutrition"]["bmr"] == 1780 and plan["inputs"] == {"weight": 80.0, "target_weight": 75.0}
    assert st.called("tabs")[0][1][0] == ("🍽️ โภชนาการ", "🏃 ตารางฝึก", "📝 หมายเหตุ")
    m = metrics(st)
    assert m["BMR"] == "1,780 kcal" and m["TDEE"].endswith("kcal") and m["เป้าหมาย/วัน"].endswith("kcal")
    htmls = [h for h, _ in st.html]
    assert len(htmls) == 2 and "kcal/วัน" in htmls[0] and "ประมาณการ" in htmls[1] and "เป้าหมาย 75.0 kg" in htmls[1]
    assert len(st.called("expander")) == 4                        # วันฝึก 4 วัน (ค่าเริ่มต้นของ slider)


def test_plan_without_projection_when_target_equals_weight():
    st = run_page(PLAN, submit=True, session={"profile": PROFILE | {"target_weight_kg": 80.0}})
    assert len(st.html) == 1                                       # มีแต่โดนัท ไม่มีกราฟประมาณการ
    assert any("ไม่แสดงกราฟ" in t for t in st.texts("caption"))


def test_plan_shows_validation_errors_for_minors():
    st = run_page(PLAN, submit=True, session={"profile": PROFILE | {"age": 16}})
    assert st.called("error") and "plan" not in st.session_state


def test_plan_uses_somatotype_from_scanner():
    st = run_page(PLAN, submit=True, session={"profile": PROFILE, "scan": {"somatotype": "endomorph"}})
    assert st.session_state["plan"]["nutrition"]["somatotype"] == "endomorph"
    assert any("ใช้ผลจากหน้า Scanner" in t for t in st.texts("caption"))


def test_plan_goal_changes_flow_through_to_workout():
    st = run_page(PLAN, submit=True, values={"เป้าหมาย": "build_muscle"}, session={"profile": PROFILE | {"target_weight_kg": 85.0}})
    assert st.session_state["plan"]["workout"]["goal"] == "build_muscle"
    assert len(st.html) == 2 and "สัปดาห์" in st.html[1][0]        # มีทั้งโดนัทและกราฟประมาณการ (น้ำหนักเพิ่มสู่เป้าหมาย)


# ---------------- หน้า Tracker ----------------
def test_tracker_empty_state():
    st = run_page(TRACKER)
    assert any("ผลจะแสดงที่นี่" in t for t in st.texts("markdown")) and not st.html


def test_tracker_predicts_and_draws_zone_bar():
    st = run_page(TRACKER, submit=True, values={"เวลา (นาที)": 20.0, "ชีพจรเฉลี่ย (ครั้ง/นาที)": 110},
                  session={"profile": PROFILE | {"age": 28, "height_cm": 175.0, "weight_kg": 72.0}})
    m = metrics(st)
    assert m["แคลอรีที่เผาผลาญ"].endswith("kcal") and m["Heart Rate Zone"].startswith("Zone")
    assert len(st.html) == 1 and "110 bpm" in st.html[0][0]
    log = st.session_state["workout_log"]
    assert len(log) == 1 and log[0]["วิธีคำนวณ"] == "โมเดล ML"
    assert "last_track" in st.session_state


def test_tracker_out_of_range_falls_back_to_formula_and_warns():
    st = run_page(TRACKER, submit=True, values={"เวลา (นาที)": 60.0, "ชีพจรเฉลี่ย (ครั้ง/นาที)": 150},
                  session={"profile": PROFILE})
    assert st.session_state["workout_log"][0]["วิธีคำนวณ"] == "สูตร Keytel" and st.called("warning")


def test_tracker_log_accumulates_and_can_be_cleared():
    first = run_page(TRACKER, submit=True, session={"profile": PROFILE})
    session = dict(first.session_state)
    second = run_page(TRACKER, submit=True, session=session)
    assert len(second.session_state["workout_log"]) == 2
    assert metrics(second)["รวมทั้งหมด"].endswith("kcal")
    cleared = run_page(TRACKER, clicks=["ล้างบันทึก"], session=dict(second.session_state))
    assert cleared.session_state["workout_log"] == [] and "last_track" not in cleared.session_state
    assert cleared.called("rerun")


def test_tracker_shows_result_after_rerun_without_submit():
    first = run_page(TRACKER, submit=True, session={"profile": PROFILE})
    again = run_page(TRACKER, session=dict(first.session_state))
    assert "แคลอรีที่เผาผลาญ" in metrics(again) and len(again.html) == 1