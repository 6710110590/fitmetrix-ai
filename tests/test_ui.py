import math
import re
import xml.etree.ElementTree as ET

import pytest

from src import ui
from src.tracker import classify_zone


def parse(snippet):
    """ชิ้นส่วนที่สร้างต้องเป็น XML ที่ถูกต้อง (ปิดแท็กครบ ไม่มี & หรือ < หลุด)"""
    return ET.fromstring(f"<root>{snippet}</root>")


def svg_of(snippet):
    root = parse(snippet)
    return next(e for e in root.iter() if e.tag.endswith("svg"))


def circles(snippet):
    return [e for e in svg_of(snippet).iter() if e.tag.endswith("circle")]


MACROS = [("โปรตีน", 160, "#22D3A6", "160 g"), ("คาร์บ", 280, "#60A5FA", "280 g"), ("ไขมัน", 70, "#FBBF24", "70 g")]


# ---------- ข้อความ/ความปลอดภัย ----------
def test_text_helpers_escape_html():
    evil = '<script>alert("x")</script>'
    for out in (ui.hero_html(evil, evil, [evil]), ui.header_html(evil, evil, "<b>"), ui.note_html(evil), ui.empty_html(evil)):
        assert "<script>" not in out and "&lt;script&gt;" in out
    parse(ui.hero_html("a & b", "x < y", ["c"]))


def test_css_has_no_characters_that_break_markdown_wrapper():
    assert "</style>" not in ui.CSS and "{" in ui.CSS


# ---------- โดนัท ----------
def test_donut_is_valid_and_arcs_cover_the_ring():
    out = ui.donut_html(MACROS, "2,345", "kcal/วัน")
    cs = circles(out)
    assert len(cs) == 3
    r = float(cs[0].get("r"))
    circ = 2 * math.pi * r
    dashes = [float(c.get("stroke-dasharray").split()[0]) for c in cs]
    assert sum(dashes) == pytest.approx(circ - 3 * 3.0, abs=1.0)            # ผลรวมส่วนโค้ง = เส้นรอบวง ลบช่องว่าง 3 ช่อง
    fracs = [d / sum(dashes) for d in dashes]
    assert fracs[1] > fracs[0] > fracs[2]                                   # ตามสัดส่วน 280 > 160 > 70


def test_donut_offsets_are_consecutive():
    cs = circles(ui.donut_html(MACROS, "x"))
    offsets = [-float(c.get("stroke-dashoffset")) for c in cs]
    assert offsets[0] == 0 and offsets == sorted(offsets)


def test_donut_zero_and_negative_values():
    assert len(circles(ui.donut_html([("a", 0, "#fff", ""), ("b", 0, "#fff", "")], "0"))) == 1      # วงแหวนเปล่า
    cs = circles(ui.donut_html([("a", -5, "#111", ""), ("b", 10, "#222", "")], "x"))
    assert len(cs) == 1 and cs[0].get("stroke") == "#222"                                           # ค่าติดลบถูกข้าม


def test_donut_without_animation_has_no_animate_tags():
    assert "<animate" in ui.donut_html(MACROS, "x") and "<animate" not in ui.donut_html(MACROS, "x", animate=False)


def test_donut_escapes_labels():
    out = ui.donut_html([("<b>x</b>", 1, "#fff", "<i>")], "<t>")
    assert "<b>x</b>" not in out and "<t>" not in out
    parse(out)


# ---------- แถบ Zone ----------
def test_zone_x_is_monotonic_and_clamped():
    xs = [ui.zone_x(p) for p in (0.1, 0.4, 0.5, 0.75, 1.0, 1.1, 2.0)]
    assert xs == sorted(xs) and xs[0] == xs[1] and xs[-1] == xs[-2]
    assert ui.zone_x(0.4) == 14 and ui.zone_x(1.1) == 560 - 14


def test_zone_bar_labels_match_hr_ranges_and_marker():
    out = ui.zone_bar_html(150, 187, classify_zone(150, 30)["zone"])
    parse(out)
    assert "112-131" in out and "168-187" in out                            # ช่วงชีพจรโซน 2 และ 5 ของ HRmax 187
    assert "150 bpm (80% HRmax)" in out
    assert out.count('stroke="#FFFFFF"') == 1                               # ไฮไลต์เฉพาะโซนที่ใช้งาน


@pytest.mark.parametrize("zone", [0, 1, 2, 3, 4, 5])
def test_zone_bar_highlights_exactly_one_segment(zone):
    out = ui.zone_bar_html(120, 187, zone)
    assert len(re.findall(r'<rect[^>]*stroke="#FFFFFF"', out)) == 1


def test_zone_bar_marker_position_follows_heart_rate():
    def marker_x(hr):
        m = re.search(r'<path d="M([\d.]+),', ui.zone_bar_html(hr, 187, 3))
        return float(m.group(1))

    assert marker_x(100) < marker_x(130) < marker_x(170)


def test_zone_bar_label_stays_inside_the_canvas():
    for hr in (40, 60, 187, 200, 230):
        out = ui.zone_bar_html(hr, 187, 3, width=560)
        x = float(re.search(r'<text x="([\d.]+)" y="\d+" text-anchor="middle" font-size="13"', out).group(1))
        assert 84 <= x <= 560 - 84


# ---------- ประมาณการน้ำหนัก ----------
def test_projection_loss_stops_at_target():
    pts = ui.projection_points(80, 75, -0.376)
    assert pts[0] == (0, 80) and pts[-1] == (14, 75.0) and len(pts) == 15           # ceil(5/0.376) = 14 สัปดาห์
    ws = [w for _, w in pts]
    assert ws == sorted(ws, reverse=True) and min(ws) == 75


def test_projection_gain_stops_at_target():
    pts = ui.projection_points(60, 66, 0.25)
    assert pts[-1] == (24, 66.0) and max(w for _, w in pts) == 66


def test_projection_capped_by_max_weeks():
    pts = ui.projection_points(100, 60, -0.3, max_weeks=26)
    assert pts[-1][0] == 26 and pts[-1][1] > 60


@pytest.mark.parametrize("w, t, c", [(80, 75, 0), (80, 80, -0.3), (80, 85, -0.3), (80, 75, 0.3)])
def test_projection_empty_when_nothing_to_project(w, t, c):
    assert ui.projection_points(w, t, c) == []


def test_projection_chart_valid_and_labelled_as_estimate():
    pts = ui.projection_points(80, 75, -0.376)
    out = ui.projection_html(pts, 75)
    parse(out)
    assert "ประมาณการ" in out and "เป้าหมาย 75.0 kg" in out and "สัปดาห์ 14" in out
    path = next(e for e in svg_of(out).iter() if e.tag.endswith("path"))
    ys = [float(s.split(",")[1]) for s in re.findall(r"[\d.]+,[\d.]+", path.get("d"))]
    assert ys == sorted(ys)                                                         # น้ำหนักลด = เส้นไหลลง (y ในจอเพิ่ม)


def test_projection_chart_with_too_few_points_shows_notice():
    out = ui.projection_html([], 70)
    parse(out)
    assert "ไม่มีข้อมูลพอ" in out and "<path" not in out