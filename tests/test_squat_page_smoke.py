"""รันหน้า pages/4_Squat.py ด้วย Streamlit จำลอง บนสำเนาในโฟลเดอร์ชั่วคราว (ไม่แตะ logs/ จริงของโปรเจกต์)"""
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from src.squat import SquatConfig
from tests.fake_streamlit import run_page
from src.ui import rep_depth_pct

REAL_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "pages").mkdir()
    shutil.copy(REAL_ROOT / "pages" / "4_Squat.py", tmp_path / "pages" / "4_Squat.py")
    (tmp_path / "src").symlink_to(REAL_ROOT / "src", target_is_directory=True)
    return tmp_path


def write_log(root: Path, name: str, view="front", with_config=True) -> None:
    (root / "logs").mkdir(exist_ok=True)
    events = [
        {"index": 1, "counted": True, "min_signal": 0.18, "duration": 2.0, "issues": []},
        {"index": 2, "counted": True, "min_signal": 0.35, "duration": 1.4, "issues": ["shallow", "tilt"]},
        {"index": None, "counted": False, "min_signal": 0.62, "duration": 0.9, "issues": ["shallow_nocount"]},
    ]
    data = {"created_at": "2026-10-07T09:30:00", "view": view, "reps": 2, "attempts": 3,
            "issue_counts": {"shallow": 1, "tilt": 1, "shallow_nocount": 1}, "events": events}
    if with_config:
        data["config"] = asdict(SquatConfig(view=view))
    (root / "logs" / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def page(root):
    return root / "pages" / "4_Squat.py"


def test_page_without_logs_shows_demo_and_empty_state(sandbox):
    st = run_page(page(sandbox))
    assert st.called("set_page_config")[0][2]["layout"] == "wide"
    assert len(st.html) == 1 and "polyline" in st.html[0][0]               # ภาพคนเส้นอย่างเดียว
    assert any("ยังไม่มีข้อมูล" in t for t in st.texts("markdown"))
    assert not st.called("metric")


def test_page_with_log_shows_stats_and_chart(sandbox):
    write_log(sandbox, "squat_20261007_093000.json")
    st = run_page(page(sandbox))
    m = {c[1][0]: c[1][1] for c in st.called("metric")}
    assert m["นับได้"] == "2 ครั้ง" and m["ครั้งที่ไม่มีข้อเตือน"] == "1 ครั้ง"
    _, _, count_thr, good_thr = SquatConfig(view="front").thresholds()
    expected = (rep_depth_pct(0.18, count_thr, good_thr) + rep_depth_pct(0.35, count_thr, good_thr)) / 2
    assert m["ลึกเฉลี่ย"] == f"{expected:.0f}%"
    assert m["เวลาต่อครั้งเฉลี่ย"] == "1.7 วินาที"
    assert len(st.html) == 2 and "ลึกถึงเกณฑ์ดี" in st.html[1][0]
    rows = st.called("table")[0][1][0]
    assert len(rows) == 3 and rows[2]["นับ"] == "ไม่นับ" and "สะโพกเอียง" in rows[1]["ข้อเตือน"]


def test_page_old_log_without_config_still_renders(sandbox):
    write_log(sandbox, "squat_20261007_093000.json", with_config=False)
    st = run_page(page(sandbox))
    assert len(st.html) == 2 and st.called("metric")


def test_page_side_view_uses_side_demo_and_captions(sandbox):
    st = run_page(page(sandbox), values={"มุมกล้อง": "side"})
    assert "มุมด้านข้าง" in st.html[0][0]
    assert any("ส้นเท้าติดพื้น" in str(c) for c in st.called("caption"))


def test_page_multiple_logs_selects_latest_by_default(sandbox):
    write_log(sandbox, "squat_20261006_100000.json", view="front")
    write_log(sandbox, "squat_20261007_093000.json", view="side")
    st = run_page(page(sandbox))
    assert any("squat_20261007_093000.json" in str(c) for c in st.called("caption"))


def test_page_unknown_issue_code_does_not_crash(sandbox):
    write_log(sandbox, "squat_20261007_093000.json")
    p = sandbox / "logs" / "squat_20261007_093000.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["events"][0]["issues"] = ["code_from_future"]
    d["issue_counts"]["code_from_future"] = 1
    p.write_text(json.dumps(d), encoding="utf-8")
    st = run_page(page(sandbox))
    assert st.called("table")