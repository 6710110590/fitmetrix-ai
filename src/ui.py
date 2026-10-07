"""
ui.py - ชิ้นส่วนหน้าตาที่ใช้ร่วมกันทุกหน้า (ธีม CSS, หัวหน้า, กราฟ SVG)

ออกแบบให้ "ไม่เพิ่ม dependency" และทดสอบได้โดยไม่ต้องมี Streamlit:
  - กราฟทุกชนิด (โดนัท, แถบ Zone ชีพจร, กราฟประมาณการน้ำหนัก) เป็นฟังก์ชันที่คืนสตริง HTML/SVG
  - แสดงผลบนหน้าเว็บด้วย show_html() (components.html) ซึ่งไม่ถูกกรอง SVG และรันแอนิเมชันได้
  - import streamlit เฉพาะในฟังก์ชันที่ต้องใช้จริง (inject_css, show_html)

กราฟประมาณการน้ำหนักเป็น "ค่าประมาณ" ที่สมมติอัตราคงที่ ไม่ใช่การพยากรณ์ ติดป้ายกำกับไว้บนกราฟเสมอ
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Optional, Sequence

PALETTE = {
    "bg": "#0B1220", "card": "#141C2E", "border": "#26324A", "text": "#E6EAF2", "muted": "#9AA6BF",
    "primary": "#22D3A6", "protein": "#22D3A6", "carbs": "#60A5FA", "fat": "#FBBF24",
    "good": "#34D399", "warn": "#FBBF24", "bad": "#F87171",
}
ZONE_COLORS = ["#60A5FA", "#34D399", "#FBBF24", "#FB923C", "#F87171"]
FONT_STACK = "'Noto Sans Thai','Leelawadee UI',Tahoma,'Segoe UI',system-ui,sans-serif"

CSS = """
.block-container{padding-top:2.2rem;padding-bottom:3rem;}
div[data-testid="stMetric"]{background:#141C2E;border:1px solid #26324A;border-radius:14px;padding:14px 18px;box-shadow:0 6px 18px rgba(0,0,0,.28);}
div[data-testid="stMetricLabel"]{color:#9AA6BF;}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px;}
div[data-testid="stExpander"]{border-radius:12px;}
button[kind="primary"],button[kind="primaryFormSubmit"]{border-radius:10px;font-weight:600;}
.fm-hero{background:linear-gradient(135deg,rgba(34,211,166,.20),rgba(96,165,250,.12) 60%,rgba(20,28,46,0));border:1px solid #26324A;border-radius:20px;padding:28px 32px;margin-bottom:18px;box-shadow:0 10px 30px rgba(0,0,0,.25);}
.fm-hero h1{margin:0 0 6px 0;font-size:2.1rem;line-height:1.2;}
.fm-hero p{margin:0;color:#B7C2D9;font-size:1.02rem;}
.fm-chips{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px;}
.fm-chip{background:rgba(34,211,166,.12);border:1px solid rgba(34,211,166,.35);color:#7FF0D1;border-radius:999px;padding:3px 12px;font-size:.82rem;}
.fm-pagehead{margin-bottom:10px;}
.fm-pagehead h2{margin:0;font-size:1.7rem;}
.fm-pagehead p{margin:4px 0 0 0;color:#9AA6BF;}
.fm-note{border-left:3px solid #FBBF24;background:rgba(251,191,36,.08);padding:10px 14px;border-radius:8px;color:#E6EAF2;font-size:.92rem;margin:8px 0;}
.fm-empty{border:1px dashed #26324A;border-radius:16px;padding:42px 24px;text-align:center;color:#9AA6BF;}
""".strip()


# ----------------------------- เมนูซ้าย (sidebar) ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# (ไฟล์หน้า, ชื่อที่แสดง, ไอคอน) เรียงตามลำดับที่แสดง แก้ชื่อที่นี่ที่เดียว (ไอคอนว่าง = ไม่แสดงไอคอน)
NAV_ITEMS = (
    ("app.py", "Profile", ""),
    ("pages/1_Profile_Scanner.py", "Scanner", ""),
    ("pages/2_Plan.py", "Plan", ""),
    ("pages/3_Tracker.py", "Tracker", ""),
    ("pages/4_Squat.py", "Squat", ""),
)

SIDEBAR_CSS = """
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0F1830 0%,#0B1220 70%);border-right:1px solid #1E2A44;}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding-top:.4rem;}
.fm-brand{display:flex;align-items:center;gap:12px;padding:4px 4px 16px 4px;margin-bottom:6px;border-bottom:1px solid #1E2A44;}
.fm-logo{width:44px;height:44px;border-radius:13px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.35rem;color:#06281F;background:linear-gradient(135deg,#22D3A6,#60A5FA);box-shadow:0 6px 18px rgba(34,211,166,.35);}
.fm-brand b{display:block;font-size:1.12rem;line-height:1.15;letter-spacing:.01em;}
.fm-brand span{display:block;color:#9AA6BF;font-size:.76rem;margin-top:2px;}
.fm-navlabel{color:#6B7893;font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;margin:10px 8px 8px 8px;}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]{justify-content:flex-start;min-height:0;border-radius:12px;padding:.55rem .85rem;margin-bottom:5px;border:1px solid transparent;transition:background .18s,border-color .18s,transform .18s;}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover{background:rgba(34,211,166,.10);border-color:rgba(34,211,166,.28);transform:translateX(3px);}
.fm-sidefoot{margin-top:22px;padding:12px 12px;border:1px solid #1E2A44;border-radius:12px;background:rgba(20,28,46,.6);color:#9AA6BF;font-size:.76rem;line-height:1.5;}
.fm-sidefoot .fm-chips{margin:0 0 8px 0;}
.fm-sidefoot .fm-chip{font-size:.7rem;padding:2px 9px;}
@media (prefers-reduced-motion:reduce){section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]{transition:none;}section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover{transform:none;}}
""".strip()


def sidebar_brand_html() -> str:
    return ('<div class="fm-brand"><div class="fm-logo">F</div>'
            '<div><b>FitMetrix AI</b><span>Fitness · Nutrition · Pose AI</span></div></div>')


def sidebar_footer_html() -> str:
    chips = "".join(f'<span class="fm-chip">{html.escape(c)}</span>' for c in ("MediaPipe", "scikit-learn"))
    return (f'<div class="fm-sidefoot"><div class="fm-chips">{chips}</div>'
            'เป็นคำแนะนำทั่วไป ไม่ใช่คำแนะนำทางการแพทย์</div>')


# ----------------------------- ส่วนที่ใช้ Streamlit ---------------------------
def inject_css(sidebar: bool = True) -> None:
    """ใส่ CSS ธีมให้หน้านั้น (เรียกหลัง st.set_page_config) และวาดเมนูซ้ายแบบกำหนดเอง (ปิดได้ด้วย sidebar=False)"""
    import streamlit as st

    st.markdown(f"<style>{CSS}\n{SIDEBAR_CSS}</style>", unsafe_allow_html=True)
    if sidebar:
        render_sidebar()


def render_sidebar() -> None:
    """
    เมนูซ้ายแบบกำหนดเอง: โลโก้ + ลิงก์ทุกหน้าพร้อมไอคอน + หมายเหตุท้ายเมนู
    ต้องปิดเมนูเดิมของ Streamlit ด้วย [client] showSidebarNavigation = false ใน .streamlit/config.toml
    ใช้ st.page_link (เปลี่ยนหน้าแล้ว session_state ไม่หาย และไฮไลต์หน้าปัจจุบันให้เอง)
    ข้ามหน้าที่ไม่มีไฟล์อยู่จริง กัน st.page_link ล้มเมื่อเปลี่ยนชื่อ/ลบไฟล์
    """
    import streamlit as st

    st.sidebar.markdown(sidebar_brand_html(), unsafe_allow_html=True)
    st.sidebar.markdown('<div class="fm-navlabel">เมนู</div>', unsafe_allow_html=True)
    for path, label, icon in NAV_ITEMS:
        if (PROJECT_ROOT / path).exists():
            st.sidebar.page_link(path, label=label, icon=icon or None)
    st.sidebar.markdown(sidebar_footer_html(), unsafe_allow_html=True)


def show_html(snippet: str, height: int) -> None:
    """แสดง HTML/SVG ใน iframe (ไม่ถูกกรอง และแอนิเมชัน SVG ทำงานได้)"""
    import streamlit.components.v1 as components

    components.html(snippet, height=height)


# ----------------------------- ชิ้นส่วนข้อความ -------------------------------
def hero_html(title: str, subtitle: str, chips: Sequence[str] = ()) -> str:
    chip_html = "".join(f'<span class="fm-chip">{html.escape(c)}</span>' for c in chips)
    return (f'<div class="fm-hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>'
            + (f'<div class="fm-chips">{chip_html}</div>' if chip_html else "") + "</div>")


def header_html(title: str, subtitle: str = "", icon: str = "") -> str:
    head = f"{html.escape(icon)} {html.escape(title)}".strip()
    return (f'<div class="fm-pagehead"><h2>{head}</h2>'
            + (f"<p>{html.escape(subtitle)}</p>" if subtitle else "") + "</div>")


def note_html(text: str) -> str:
    return f'<div class="fm-note">{html.escape(text)}</div>'


def empty_html(text: str) -> str:
    return f'<div class="fm-empty">{html.escape(text)}</div>'


def _doc(body: str) -> str:
    """ครอบชิ้นส่วนที่จะแสดงใน iframe ให้พื้นหลังโปร่งใสและใช้ฟอนต์/สีเดียวกับธีม"""
    return (f'<meta charset="utf-8"/><style>body{{margin:0;background:transparent;font-family:{FONT_STACK};color:{PALETTE["text"]};}}</style>'
            f"{body}")


# ----------------------------- โดนัท (สัดส่วน Macros) ------------------------
def donut_html(segments: Sequence[tuple], center_title: str, center_sub: str = "", size: int = 210,
               thickness: int = 26, animate: bool = True) -> str:
    """
    segments: [(ชื่อ, ค่า, สี, ข้อความรายละเอียดในคำอธิบาย), ...] ค่าติดลบนับเป็น 0
    แต่ละส่วนยาวตามสัดส่วนของผลรวม เริ่มวาดจากด้านบนตามเข็มนาฬิกา
    """
    cx = cy = size / 2
    r = (size - thickness) / 2 - 2
    circ = 2 * math.pi * r
    values = [max(float(s[1]), 0.0) for s in segments]
    total = sum(values)
    positive = sum(1 for v in values if v > 0)
    gap = 3.0 if positive > 1 else 0.0

    arcs, acc = [], 0.0
    if total <= 0:
        arcs.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="none" stroke="{PALETTE["border"]}" stroke-width="{thickness}"/>')
    else:
        for (label, _v, color, _d), v in zip(segments, values):
            if v <= 0:
                continue
            length = circ * v / total
            dash = max(length - gap, 0.5)
            anim = (f'<animate attributeName="stroke-dasharray" from="0 {circ:.2f}" to="{dash:.2f} {circ - dash:.2f}" '
                    'dur="0.9s" begin="0s" fill="freeze"/>') if animate else ""
            arcs.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="none" stroke="{color}" stroke-width="{thickness}" '
                f'stroke-dasharray="{dash:.2f} {circ - dash:.2f}" stroke-dashoffset="{-acc:.2f}" '
                f'transform="rotate(-90 {cx} {cy})" data-label="{html.escape(str(label))}">{anim}</circle>')
            acc += length

    svg = (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">'
           + "".join(arcs)
           + f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="{size * 0.15:.0f}" font-weight="700" '
             f'fill="{PALETTE["text"]}">{html.escape(center_title)}</text>'
           + f'<text x="{cx}" y="{cy + size * 0.11:.0f}" text-anchor="middle" font-size="{size * 0.065:.0f}" '
             f'fill="{PALETTE["muted"]}">{html.escape(center_sub)}</text></svg>')
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin:7px 0;">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{color};display:inline-block;"></span>'
        f'<span><b>{html.escape(str(label))}</b><br/><span style="color:{PALETTE["muted"]};font-size:13px;">'
        f"{html.escape(str(detail))}</span></span></div>"
        for label, _v, color, detail in segments)
    return _doc(f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:26px;">{svg}<div>{legend}</div></div>')


# ----------------------------- แถบ Zone ชีพจร --------------------------------
ZONE_SCALE = (0.40, 1.10)       # ช่วง %HRmax ที่แสดงบนแถบ


def zone_x(pct: float, width: float = 560, margin: float = 14) -> float:
    """ตำแหน่ง x ของ %HRmax บนแถบ (ถูกจำกัดให้อยู่ในช่วงที่แสดง)"""
    lo, hi = ZONE_SCALE
    p = min(max(pct, lo), hi)
    return margin + (p - lo) / (hi - lo) * (width - 2 * margin)


def zone_bar_html(heart_rate: float, hr_max: float, active_zone: int, width: int = 560, animate: bool = True) -> str:
    """แถบ 5 โซนตามช่วง %HRmax พร้อมตัวชี้ชีพจรปัจจุบัน active_zone = 0 (ต่ำกว่าโซน 1) ถึง 5"""
    from src.tracker import ZONES

    pct = heart_rate / hr_max if hr_max else 0.0
    top, bar_h, m = 38, 26, 14
    parts = []

    def rect(p0, p1, color, opacity, label=None, outline=False):
        x0, x1 = zone_x(p0, width, m), zone_x(p1, width, m)
        stroke = ' stroke="#FFFFFF" stroke-width="2"' if outline else ""
        parts.append(f'<rect x="{x0:.1f}" y="{top}" width="{x1 - x0 - 2:.1f}" height="{bar_h}" rx="6" fill="{color}" '
                     f'opacity="{opacity}"{stroke}/>')
        if label:
            parts.append(f'<text x="{(x0 + x1) / 2 - 1:.1f}" y="{top + bar_h + 18}" text-anchor="middle" font-size="12" '
                         f'fill="{PALETTE["text"]}" font-weight="600">{html.escape(label[0])}</text>'
                         f'<text x="{(x0 + x1) / 2 - 1:.1f}" y="{top + bar_h + 33}" text-anchor="middle" font-size="11" '
                         f'fill="{PALETTE["muted"]}">{html.escape(label[1])}</text>')

    rect(ZONE_SCALE[0], 0.50, PALETTE["border"], 0.9 if active_zone == 0 else 0.5, ("พัก", "<50%"), active_zone == 0)
    for (num, _name, lo, hi, _desc), color in zip(ZONES, ZONE_COLORS):
        is_active = active_zone == num
        rect(lo, hi, color, 1.0 if is_active else 0.4,
             (f"Z{num}", f"{round(hr_max * lo)}-{round(hr_max * hi)}"), is_active)
    rect(1.00, ZONE_SCALE[1], "#7F1D1D", 0.7, None)

    mx = zone_x(pct, width, m)
    slide = (f'<animateTransform attributeName="transform" type="translate" from="-30 0" to="0 0" dur="0.6s" '
             'begin="0s" fill="freeze"/>') if animate else ""
    label_x = min(max(mx, 84), width - 84)                     # ป้ายยาวราว 150px ต้องไม่ล้นขอบ
    marker = (f'<g>{slide}<path d="M{mx - 8:.1f},{top - 12} L{mx + 8:.1f},{top - 12} L{mx:.1f},{top - 2} Z" fill="#FFFFFF"/>'
              f'<text x="{label_x:.1f}" y="{top - 18}" text-anchor="middle" font-size="13" font-weight="700" '
              f'fill="#FFFFFF">{heart_rate:.0f} bpm ({pct * 100:.0f}% HRmax)</text></g>')
    height = top + bar_h + 44
    return _doc(f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;height:auto" '
                'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + marker + "</svg>")


# ----------------------------- กราฟประมาณการน้ำหนัก --------------------------
def projection_points(weight_kg: float, target_kg: float, weekly_change_kg: float, max_weeks: int = 26) -> list[tuple[int, float]]:
    """
    ประมาณน้ำหนักรายสัปดาห์ โดยสมมติอัตราเปลี่ยนคงที่ (ไม่คิดว่า TDEE เปลี่ยนตามน้ำหนัก) หยุดที่น้ำหนักเป้าหมาย
    คืน [] เมื่อไม่มีอะไรจะประมาณ (ไม่เปลี่ยน, สวนทางกับเป้า, หรือเป้าเท่าปัจจุบัน)
    """
    diff = target_kg - weight_kg
    if weekly_change_kg == 0 or diff == 0 or diff * weekly_change_kg < 0:
        return []
    weeks_needed = math.ceil(abs(diff / weekly_change_kg))
    n = min(weeks_needed, max_weeks)
    pts = []
    for t in range(n + 1):
        w = weight_kg + weekly_change_kg * t
        w = max(w, target_kg) if weekly_change_kg < 0 else min(w, target_kg)
        pts.append((t, round(w, 2)))
    return pts


def projection_html(points: Sequence[tuple[int, float]], target_kg: float, width: int = 620, height: int = 270,
                    animate: bool = True) -> str:
    """กราฟเส้นประมาณการน้ำหนัก (SVG) พร้อมเส้นเป้าหมายและป้าย 'ประมาณการ'"""
    if len(points) < 2:
        return _doc(empty_html("ไม่มีข้อมูลพอสำหรับประมาณการ"))
    ml, mr, mt, mb = 56, 28, 34, 40
    pw, ph = width - ml - mr, height - mt - mb
    weeks = [p[0] for p in points]
    ws = [p[1] for p in points] + [target_kg]
    lo, hi = min(ws), max(ws)
    pad = max((hi - lo) * 0.12, 0.5)
    lo, hi = lo - pad, hi + pad

    def X(t):
        return ml + (t - weeks[0]) / max(weeks[-1] - weeks[0], 1) * pw

    def Y(w):
        return mt + (hi - w) / (hi - lo) * ph

    grid = []
    for i in range(5):
        gv = lo + (hi - lo) * i / 4
        gy = Y(gv)
        grid.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml + pw}" y2="{gy:.1f}" stroke="{PALETTE["border"]}" stroke-width="1"/>'
                    f'<text x="{ml - 8}" y="{gy + 4:.1f}" text-anchor="end" font-size="11" fill="{PALETTE["muted"]}">{gv:.1f}</text>')
    xticks = sorted({weeks[0], weeks[len(weeks) // 2], weeks[-1]})
    for t in xticks:
        grid.append(f'<text x="{X(t):.1f}" y="{mt + ph + 20}" text-anchor="middle" font-size="11" '
                    f'fill="{PALETTE["muted"]}">สัปดาห์ {t}</text>')

    d = "M" + " L".join(f"{X(t):.1f},{Y(w):.1f}" for t, w in points)
    anim = ('<animate attributeName="stroke-dashoffset" from="100" to="0" dur="1.2s" begin="0s" fill="freeze"/>'
            if animate else "")
    line = (f'<path d="{d}" fill="none" stroke="{PALETTE["primary"]}" stroke-width="3" stroke-linecap="round" '
            f'stroke-linejoin="round" pathLength="100" stroke-dasharray="100" stroke-dashoffset="0">{anim}</path>')
    ty = Y(target_kg)
    target = (f'<line x1="{ml}" y1="{ty:.1f}" x2="{ml + pw}" y2="{ty:.1f}" stroke="{PALETTE["warn"]}" stroke-width="1.5" '
              f'stroke-dasharray="6 5"/><text x="{ml + 8}" y="{ty - 6:.1f}" text-anchor="start" font-size="12" '
              f'fill="{PALETTE["warn"]}">เป้าหมาย {target_kg:.1f} kg</text>')
    (t0, w0), (t1, w1) = points[0], points[-1]
    dots = (f'<circle cx="{X(t0):.1f}" cy="{Y(w0):.1f}" r="5" fill="{PALETTE["primary"]}"/>'
            f'<text x="{X(t0) + 8:.1f}" y="{Y(w0) - 8:.1f}" font-size="12" fill="{PALETTE["text"]}">{w0:.1f} kg</text>'
            f'<circle cx="{X(t1):.1f}" cy="{Y(w1):.1f}" r="5" fill="{PALETTE["primary"]}"/>'
            f'<text x="{X(t1) - 8:.1f}" y="{Y(w1) + (18 if w1 < w0 else -10):.1f}" text-anchor="end" font-size="12" '
            f'fill="{PALETTE["text"]}">{w1:.1f} kg (สัปดาห์ {t1})</text>')
    badge = (f'<text x="{ml}" y="18" font-size="12" fill="{PALETTE["muted"]}">ประมาณการ (สมมติอัตราคงที่ ไม่ใช่การพยากรณ์)</text>')
    return _doc(f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;height:auto" '
                'xmlns="http://www.w3.org/2000/svg">' + badge + "".join(grid) + target + line + dots + "</svg>")


# ============================ หน้า Squat (ขั้น 5) ================================
# กราฟรายครั้ง + ภาพคนเส้นสาธิตท่า เป็น SVG ที่สร้างจากโค้ด ไม่เพิ่ม dependency และทดสอบได้โดยไม่ต้องมี Streamlit

def rep_depth_pct(min_signal: float, count: float, good: float) -> float:
    """
    ความลึกของครั้งนั้นเทียบกับเกณฑ์ (ค่าสัญญาณยิ่งน้อย = ยิ่งลึก)
    0 = เพิ่งถึงเกณฑ์นับ, 100 = ลึกถึงเกณฑ์ 'ดี', ติดลบ = ไม่ถึงเกณฑ์นับ
    เป็นการเทียบกับเกณฑ์ตั้งต้นที่ยังไม่ได้สอบเทียบ ไม่ใช่คะแนนคุณภาพท่าที่แท้จริง
    """
    span = count - good
    if span <= 1e-9:
        return 100.0 if min_signal <= good else 0.0
    return (count - min_signal) / span * 100.0


def rep_status(event: dict) -> str:
    """'good' = นับและไม่มีข้อเตือน | 'warn' = นับแต่มีข้อเตือน | 'miss' = ไม่นับ"""
    if not event.get("counted"):
        return "miss"
    return "warn" if event.get("issues") else "good"


def rep_stats(events: Sequence[dict], count: float, good: float) -> dict:
    """สถิติสรุปของเซสชัน (ไม่มีครั้งที่นับ -> ค่าเฉลี่ยเป็น None)"""
    counted = [e for e in events if e.get("counted")]
    return {
        "good_reps": sum(1 for e in counted if not e.get("issues")),
        "counted": len(counted),
        "avg_depth_pct": (sum(rep_depth_pct(e["min_signal"], count, good) for e in counted) / len(counted)) if counted else None,
        "avg_duration": (sum(e["duration"] for e in counted) / len(counted)) if counted else None,
    }


REP_COLORS = {"good": PALETTE["good"], "warn": PALETTE["warn"], "miss": PALETTE["bad"]}


def rep_chart_html(events: Sequence[dict], count: float, good: float, issue_labels: Optional[dict] = None,
                   width: int = 640, height: int = 280, max_reps: int = 20, animate: bool = True) -> str:
    """
    แผนภูมิแท่งรายครั้ง: ความสูงแท่ง = ความลึกเทียบเกณฑ์ (100% = ลึกถึงเกณฑ์ดี) สี = ผลของครั้งนั้น
    ใต้แท่งแสดงลำดับครั้ง (ไม่นับ = '-') และเวลาที่ใช้ เอาเมาส์ชี้แท่งเพื่อดูข้อเตือนเต็ม
    แสดงล่าสุดไม่เกิน max_reps ครั้ง
    """
    if not events:
        return _doc(empty_html("ยังไม่มีข้อมูลรายครั้ง"))
    issue_labels = issue_labels or {}
    shown = list(events)[-max_reps:]
    n = len(shown)
    ml, mr, mt, mb = 46, 16, 40, 52
    pw, ph = width - ml - mr, height - mt - mb
    top_v = 130.0                                            # แกนตั้งแสดง 0..130%

    def Y(v: float) -> float:
        return mt + (top_v - min(max(v, 0.0), top_v)) / top_v * ph

    slot = pw / n
    bw = min(44.0, slot * 0.62)
    parts = []
    for v, label in ((0, "0%"), (50, "50%"), (100, "100%")):
        parts.append(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml + pw}" y2="{Y(v):.1f}" stroke="{PALETTE["border"]}" stroke-width="1"/>'
                     f'<text x="{ml - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="{PALETTE["muted"]}">{label}</text>')
    parts.append(f'<line x1="{ml}" y1="{Y(100):.1f}" x2="{ml + pw}" y2="{Y(100):.1f}" stroke="{PALETTE["primary"]}" '
                 'stroke-width="1.5" stroke-dasharray="6 5"/>'
                 f'<text x="{ml + pw}" y="{Y(100) - 6:.1f}" text-anchor="end" font-size="11" fill="{PALETTE["primary"]}">ลึกถึงเกณฑ์ดี</text>')

    for i, e in enumerate(shown):
        status = rep_status(e)
        color = REP_COLORS[status]
        depth = rep_depth_pct(e["min_signal"], count, good)
        bar_h = max(Y(0) - Y(max(depth, 0.0)), 5.0)          # ไม่นับ/ตื้นมาก ยังเห็นเป็นแท่งเตี้ยๆ
        x = ml + slot * i + (slot - bw) / 2
        y = Y(0) - bar_h
        issues = ", ".join(issue_labels.get(c, str(c)) for c in e.get("issues", [])) or "ไม่มีข้อเตือน"
        head = f"ครั้งที่ {e['index']}" if e.get("counted") else "ไม่นับ"
        tip = html.escape(f"{head}: ลึก {depth:.0f}% ของเกณฑ์ดี · {e['duration']:.1f} วินาที · {issues}")
        anim = (f'<animate attributeName="height" from="0" to="{bar_h:.1f}" dur="0.6s" begin="0s" fill="freeze"/>'
                f'<animate attributeName="y" from="{Y(0):.1f}" to="{y:.1f}" dur="0.6s" begin="0s" fill="freeze"/>') if animate else ""
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" rx="5" fill="{color}"><title>{tip}</title>{anim}</rect>')
        cx = x + bw / 2
        parts.append(f'<text x="{cx:.1f}" y="{Y(0) + 17}" text-anchor="middle" font-size="12" font-weight="600" '
                     f'fill="{PALETTE["text"]}">{e["index"] if e.get("counted") else "-"}</text>'
                     f'<text x="{cx:.1f}" y="{Y(0) + 32}" text-anchor="middle" font-size="10" '
                     f'fill="{PALETTE["muted"]}">{e["duration"]:.1f}s</text>')

    legend_items = (("good", "ท่าดี"), ("warn", "นับแต่มีข้อเตือน"), ("miss", "ไม่นับ"))
    lx, legend = ml, []
    for key, text in legend_items:
        legend.append(f'<rect x="{lx}" y="9" width="11" height="11" rx="3" fill="{REP_COLORS[key]}"/>'
                      f'<text x="{lx + 16}" y="19" font-size="12" fill="{PALETTE["text"]}">{text}</text>')
        lx += 16 + 8 * len(text) + 20
    if len(events) > n:
        legend.append(f'<text x="{ml + pw}" y="19" text-anchor="end" font-size="11" fill="{PALETTE["muted"]}">แสดง {n} ครั้งล่าสุดจาก {len(events)}</text>')
    return _doc(f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;height:auto" '
                'xmlns="http://www.w3.org/2000/svg">' + "".join(legend) + "".join(parts) + "</svg>")


# ภาพคนเส้นสาธิตท่า Squat: (ชื่อส่วน, จุดตอนยืน, จุดตอนก้นท่า) พิกัดในกรอบ 200x260 พื้นอยู่ที่ y=240
# สัดส่วนต้นขา/หน้าแข้ง/ลำตัวคงที่ในมุมข้าง ส่วนมุมหน้าหดตามการมองเห็น (foreshortening)
_DEMO_POSES = {
    "side": {
        "limbs": [
            ("leg",   [(100, 240), (100, 172), (100, 104)], [(100, 240), (128, 178), (62, 184)]),
            ("foot",  [(100, 240), (122, 240)],             [(100, 240), (122, 240)]),
            ("torso", [(100, 104), (100, 46)],              [(62, 184), (95, 136)]),
            ("arm",   [(100, 46), (102, 98)],               [(95, 136), (146, 132)]),
        ],
        "head": ((100, 24), (109, 116)),
        "joints": ["leg", "torso", "arm"],
    },
    "front": {
        "limbs": [
            ("legL",  [(80, 240), (84, 174), (86, 106)],    [(78, 240), (66, 180), (86, 176)]),
            ("legR",  [(120, 240), (116, 174), (114, 106)], [(122, 240), (134, 180), (114, 176)]),
            ("hips",  [(86, 106), (114, 106)],              [(86, 176), (114, 176)]),
            ("torso", [(100, 106), (100, 50)],              [(100, 176), (100, 122)]),
            ("shld",  [(72, 50), (128, 50)],                [(72, 122), (128, 122)]),
            ("armL",  [(72, 50), (66, 104)],                [(72, 122), (88, 130)]),
            ("armR",  [(128, 50), (134, 104)],              [(128, 122), (112, 130)]),
        ],
        "head": ((100, 24), (100, 96)),
        "joints": ["legL", "legR", "armL", "armR"],
    },
}
DEMO_CAPTIONS = {
    "side": "ด้านข้าง: ส้นเท้าติดพื้น ลำตัวเอียงพอประมาณ อกขึ้น ลงจนต้นขาใกล้ขนานพื้น",
    "front": "ด้านหน้า: เข่าชี้ไปทางเดียวกับปลายเท้า ไม่หุบเข้า สะโพกสองข้างอยู่ระดับเดียวกัน",
}


def _pts(points: Sequence[tuple]) -> str:
    return " ".join(f"{x},{y}" for x, y in points)


def squat_demo_html(view: str = "side", animate: bool = True, period: float = 3.4) -> str:
    """ภาพคนเส้นสาธิตท่า Squat (SVG เคลื่อนไหวต่อเนื่อง) view = 'side' | 'front' วาดจากโค้ด ไม่มีปัญหาลิขสิทธิ์/ใบหน้า"""
    if view not in _DEMO_POSES:
        raise ValueError("view ต้องเป็น 'side' หรือ 'front'")
    pose = _DEMO_POSES[view]
    color, joint_color = PALETTE["primary"], "#FFFFFF"
    spline = 'calcMode="spline" keyTimes="0;0.5;1" keySplines=".45 0 .2 1;.45 0 .2 1"'

    def anim(attr: str, a, b) -> str:
        if not animate:
            return ""
        return f'<animate attributeName="{attr}" values="{a};{b};{a}" dur="{period}s" repeatCount="indefinite" {spline}/>'

    parts = [f'<line x1="14" y1="240" x2="186" y2="240" stroke="{PALETTE["border"]}" stroke-width="3" stroke-linecap="round"/>']
    for name, stand, bottom in pose["limbs"]:
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="7" stroke-linecap="round" stroke-linejoin="round" '
                     f'points="{_pts(stand)}">{anim("points", _pts(stand), _pts(bottom))}</polyline>')
    (hx0, hy0), (hx1, hy1) = pose["head"]
    parts.append(f'<circle cx="{hx0}" cy="{hy0}" r="14" fill="none" stroke="{color}" stroke-width="6">'
                 f'{anim("cx", hx0, hx1)}{anim("cy", hy0, hy1)}</circle>')
    for name, stand, bottom in pose["limbs"]:
        if name not in pose["joints"]:
            continue
        for (x0, y0), (x1, y1) in zip(stand, bottom):
            parts.append(f'<circle cx="{x0}" cy="{y0}" r="4" fill="{joint_color}">{anim("cx", x0, x1)}{anim("cy", y0, y1)}</circle>')
    label = "มุมด้านข้าง" if view == "side" else "มุมด้านหน้า"
    parts.append(f'<text x="100" y="257" text-anchor="middle" font-size="12" fill="{PALETTE["muted"]}">{label}</text>')
    return _doc('<svg viewBox="0 0 200 262" width="100%" style="max-width:260px;height:auto;display:block;margin:0 auto" '
                'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>")