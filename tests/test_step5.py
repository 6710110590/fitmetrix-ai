"""ทดสอบขั้น 5: กราฟรายครั้ง ภาพคนเส้น และกรอบสีของ live_squat (ไม่ต้องมี Streamlit/กล้อง)"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import live_squat as L
from src import ui
from src.squat import FrameResult, RepEvent


def svg_of(doc: str) -> ET.Element:
    """ดึง <svg> ออกจากเอกสารแล้ว parse เป็น XML (พังถ้าแท็กไม่ปิด/อักขระผิด)"""
    m = re.search(r"<svg.*</svg>", doc, re.S)
    assert m, "ไม่พบ svg"
    return ET.fromstring(m.group(0))


def ev(index, counted, min_signal, duration=1.5, issues=()):
    return {"index": index, "counted": counted, "min_signal": min_signal, "duration": duration, "issues": list(issues)}


# ---------- ความลึก/สถานะ/สถิติ ----------
def test_depth_pct_front_thresholds():
    assert ui.rep_depth_pct(0.50, 0.50, 0.20) == pytest.approx(0)        # เพิ่งถึงเกณฑ์นับ
    assert ui.rep_depth_pct(0.20, 0.50, 0.20) == pytest.approx(100)      # ถึงเกณฑ์ดี
    assert ui.rep_depth_pct(0.35, 0.50, 0.20) == pytest.approx(50)
    assert ui.rep_depth_pct(0.60, 0.50, 0.20) < 0                         # ไม่ถึงเกณฑ์นับ
    assert ui.rep_depth_pct(0.10, 0.50, 0.20) > 100                       # ลึกกว่าเกณฑ์ดี


def test_depth_pct_side_angle_and_degenerate():
    assert ui.rep_depth_pct(110, 120, 100) == pytest.approx(50)           # มุมเข่า ยิ่งน้อยยิ่งลึก
    assert ui.rep_depth_pct(95, 100, 100) == 100.0                         # count == good ไม่หารศูนย์
    assert ui.rep_depth_pct(105, 100, 100) == 0.0


def test_rep_status():
    assert ui.rep_status(ev(1, True, 0.2)) == "good"
    assert ui.rep_status(ev(2, True, 0.3, issues=["valgus"])) == "warn"
    assert ui.rep_status(ev(None, False, 0.6, issues=["shallow_nocount"])) == "miss"


def test_rep_stats():
    events = [ev(1, True, 0.20, 2.0), ev(2, True, 0.35, 1.0, ["shallow"]), ev(None, False, 0.6, 0.9, ["shallow_nocount"])]
    s = ui.rep_stats(events, 0.50, 0.20)
    assert s["counted"] == 2 and s["good_reps"] == 1
    assert s["avg_depth_pct"] == pytest.approx(75)
    assert s["avg_duration"] == pytest.approx(1.5)                        # ไม่รวมครั้งที่ไม่นับ


def test_rep_stats_empty_and_no_counted():
    assert ui.rep_stats([], 0.5, 0.2)["avg_depth_pct"] is None
    s = ui.rep_stats([ev(None, False, 0.6)], 0.5, 0.2)
    assert s["counted"] == 0 and s["avg_duration"] is None


# ---------- กราฟรายครั้ง ----------
def test_rep_chart_valid_svg_and_bars():
    events = [ev(1, True, 0.20), ev(2, True, 0.35, issues=["tilt"]), ev(None, False, 0.60, issues=["shallow_nocount"])]
    root = svg_of(ui.rep_chart_html(events, 0.50, 0.20, {"tilt": "สะโพกเอียง", "shallow_nocount": "ไม่นับ"}))
    ns = {"s": "http://www.w3.org/2000/svg"}
    bars = [r for r in root.findall(".//s:rect", ns) if r.find("s:title", ns) is not None]
    assert len(bars) == 3
    fills = [b.get("fill") for b in bars]
    assert fills == [ui.PALETTE["good"], ui.PALETTE["warn"], ui.PALETTE["bad"]]
    heights = [float(b.get("height")) for b in bars]
    assert heights[0] > heights[1] > heights[2] >= 5.0                    # ลึกกว่า = สูงกว่า, ไม่นับยังเห็นแท่งเตี้ย
    assert "สะโพกเอียง" in ui.rep_chart_html(events, 0.50, 0.20, {"tilt": "สะโพกเอียง"})


def test_rep_chart_bars_stay_inside_plot():
    events = [ev(i, True, 0.05) for i in range(1, 6)]                     # ลึกเกิน 100% มากๆ ต้องถูกตัดที่แกน
    root = svg_of(ui.rep_chart_html(events, 0.50, 0.20, height=280))
    ns = {"s": "http://www.w3.org/2000/svg"}
    for r in root.findall(".//s:rect", ns):
        if r.find("s:title", ns) is not None:
            assert float(r.get("y")) >= 0 and float(r.get("y")) + float(r.get("height")) <= 280


def test_rep_chart_empty_and_truncation():
    assert "ยังไม่มีข้อมูลรายครั้ง" in ui.rep_chart_html([], 0.5, 0.2)
    many = [ev(i, True, 0.3) for i in range(1, 31)]
    doc = ui.rep_chart_html(many, 0.5, 0.2, max_reps=20)
    root = svg_of(doc)
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert len([r for r in root.findall(".//s:rect", ns) if r.find("s:title", ns) is not None]) == 20
    assert "แสดง 20 ครั้งล่าสุดจาก 30" in doc


def test_rep_chart_escapes_issue_text():
    doc = ui.rep_chart_html([ev(1, True, 0.3, issues=["x"])], 0.5, 0.2, {"x": '<script>alert("a")</script>'})
    assert "<script>" not in doc
    svg_of(doc)


def test_rep_chart_no_animation_has_no_animate_tags():
    assert "<animate" not in ui.rep_chart_html([ev(1, True, 0.3)], 0.5, 0.2, animate=False)


# ---------- ภาพคนเส้น ----------
@pytest.mark.parametrize("view", ["side", "front"])
def test_demo_valid_and_animated(view):
    doc = ui.squat_demo_html(view)
    root = svg_of(doc)
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert root.findall(".//s:polyline", ns) and root.findall(".//s:circle", ns)
    assert 'repeatCount="indefinite"' in doc
    assert "<animate" not in ui.squat_demo_html(view, animate=False)


@pytest.mark.parametrize("view", ["side", "front"])
def test_demo_animation_values_are_consistent(view):
    """ทุก <animate points> ต้องมีจำนวนจุดเท่ากันทั้งสามค่า (ยืน;ก้นท่า;ยืน) ไม่งั้นเบราว์เซอร์ไม่เล่นแอนิเมชัน"""
    root = svg_of(ui.squat_demo_html(view))
    ns = {"s": "http://www.w3.org/2000/svg"}
    n = 0
    for a in root.iter("{http://www.w3.org/2000/svg}animate"):
        values = a.get("values").split(";")
        assert len(values) == 3 and values[0] == values[2]
        if a.get("attributeName") == "points":
            counts = {len(v.split()) for v in values}
            assert len(counts) == 1
            n += 1
    assert n >= 4


@pytest.mark.parametrize("view", ["side", "front"])
def test_demo_points_inside_canvas(view):
    root = svg_of(ui.squat_demo_html(view))
    for a in root.iter("{http://www.w3.org/2000/svg}animate"):
        if a.get("attributeName") == "points":
            for v in a.get("values").split(";"):
                for pt in v.split():
                    x, y = map(float, pt.split(","))
                    assert 0 <= x <= 200 and 0 <= y <= 240                 # ไม่ทะลุพื้น (y=240) และไม่หลุดกรอบ


def test_demo_side_geometry_limb_lengths_stable():
    """มุมข้าง ต้นขา/หน้าแข้งยาวพอๆ กันทั้งตอนยืนและก้นท่า (ภาพไม่ยืด/หด)"""
    import math
    limbs = {n: (s, b) for n, s, b in ui._DEMO_POSES["side"]["limbs"]}
    for pts in limbs["leg"]:
        ankle, knee, hip = pts
        shin, thigh = math.dist(ankle, knee), math.dist(knee, hip)
        assert 60 <= shin <= 75 and 60 <= thigh <= 75
    torso_stand, torso_bottom = (math.dist(*p) for p in limbs["torso"])
    assert abs(torso_stand - torso_bottom) < 6


def test_demo_rejects_bad_view():
    with pytest.raises(ValueError):
        ui.squat_demo_html("back")


# ---------- กรอบสีของกล้อง ----------
def fr(status="ready", live=()):
    return FrameResult(status=status, phase="up", reps=0, live_issues=list(live))


def rep(issues=()):
    return RepEvent(index=1, counted="shallow_nocount" not in issues, min_signal=0.2, duration=1.5, issues=list(issues))


def test_border_not_ready_is_yellow():
    for st_ in ("no_pose", "partial", "calibrating"):
        assert L.border_style(fr(st_), None, 99)[0] == L.YELLOW


def test_border_live_issue_is_red_and_thick():
    color, thick = L.border_style(fr(live=["valgus"]), None, 99)
    assert color == L.RED and thick > L.border_style(fr(), None, 99)[1]


def test_border_normal_is_thin_green():
    assert L.border_style(fr(), None, 99) == (L.GREEN, 6)


def test_border_after_rep_colors():
    assert L.border_style(fr(), rep(), 0.5) == (L.GREEN, 14)
    assert L.border_style(fr(), rep(["tilt"]), 0.5)[0] == L.YELLOW
    assert L.border_style(fr(), rep(["shallow_nocount"]), 0.5)[0] == L.RED
    assert L.border_style(fr(), rep(["shallow_nocount"]), 2.0) == (L.GREEN, 6)    # เกิน 1.5 วินาที กลับเป็นปกติ


def test_border_live_issue_beats_recent_good_rep():
    assert L.border_style(fr(live=["lean"]), rep(), 0.1)[0] == L.RED


def test_draw_border_paints_edges_only():
    frame = np.zeros((480, 640, 3), np.uint8)
    L.draw_border(frame, L.RED, 14)
    assert tuple(frame[0, 320]) == L.RED and tuple(frame[479, 320]) == L.RED
    assert tuple(frame[240, 0]) == L.RED and tuple(frame[240, 639]) == L.RED
    assert tuple(frame[240, 320]) == (0, 0, 0)                            # กลางภาพไม่ถูกแตะ
    assert tuple(frame[240, 13]) == L.RED and tuple(frame[240, 14]) == (0, 0, 0)  # หนา 14 พิกเซลพอดี


def test_draw_border_scales_with_height():
    frame = np.zeros((960, 1280, 3), np.uint8)
    L.draw_border(frame, L.GREEN, 6)
    assert tuple(frame[480, 11]) == L.GREEN and tuple(frame[480, 12]) == (0, 0, 0)  # ภาพสูงสองเท่า หนาสองเท่า


def test_run_live_draws_border_on_frames():
    """ลูปหลักต้องวาดกรอบจริง: เฟรมที่ส่งให้ show() มีสีกรอบที่มุมภาพ"""
    from src.squat import SquatAnalyzer, SquatConfig

    class Cam:
        def read(self):
            return True, np.zeros((480, 640, 3), np.uint8)

    seen = []

    def show(frame):
        seen.append(tuple(frame[0, 0]))
        return -1

    L.run_live(Cam(), lambda rgb, ts: None, SquatAnalyzer(SquatConfig()), show, None, max_frames=3)
    assert seen == [L.YELLOW] * 3                                          # ไม่พบคน = เหลือง