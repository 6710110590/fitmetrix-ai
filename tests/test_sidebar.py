"""ทดสอบเมนูซ้ายแบบกำหนดเอง (ชื่อหน้า ไอคอน การข้ามไฟล์ที่ไม่มี และรันกับ Streamlit จริง)"""
import shutil
from pathlib import Path

import pytest

from src import ui
from tests.fake_streamlit import run_page

ROOT = Path(__file__).resolve().parent.parent


def test_nav_items_shape():
    paths = [p for p, _, _ in ui.NAV_ITEMS]
    labels = [l for _, l, _ in ui.NAV_ITEMS]
    assert paths[0] == "app.py" and labels[0] == "Profile"            # แทนคำว่า app
    assert len(set(paths)) == len(paths) and len(set(labels)) == len(labels)
    assert all(p == "app.py" or (p.startswith("pages/") and p.endswith(".py")) for p in paths)


def test_nav_items_point_to_existing_pages():
    missing = [p for p, _, _ in ui.NAV_ITEMS if not (ROOT / p).exists()]
    assert not missing, f"NAV_ITEMS ชี้ไปไฟล์ที่ไม่มี (เปลี่ยนชื่อไฟล์หรือยัง?): {missing}"


def test_brand_and_footer_html():
    assert "FitMetrix AI" in ui.sidebar_brand_html()
    assert "ไม่ใช่คำแนะนำทางการแพทย์" in ui.sidebar_footer_html()


def test_sidebar_css_targets_page_links_and_is_scoped():
    css = ui.SIDEBAR_CSS
    assert 'stPageLink-NavLink' in css and 'aria-current="page"' in css and "prefers-reduced-motion" in css
    assert css.count("{") == css.count("}")


def test_pages_render_sidebar_links_for_existing_files(tmp_path):
    page = tmp_path / "pages" / "4_Squat.py"
    page.parent.mkdir()
    shutil.copy(ROOT / "pages" / "4_Squat.py", page)
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    st = run_page(page)
    linked = [c[1][0] for c in st.called("sidebar_page_link")]
    assert linked == [p for p, _, _ in ui.NAV_ITEMS if (ROOT / p).exists()]
    first = st.called("sidebar_page_link")[0]
    assert first[1][0] == "app.py" and first[2]["label"] == "Profile" and first[2]["icon"] is None
    assert any("FitMetrix AI" in t for t in st.texts("markdown"))


def test_inject_css_without_sidebar_adds_no_links(tmp_path):
    page = tmp_path / "p.py"
    page.write_text("from src.ui import inject_css\ninject_css(sidebar=False)\n", encoding="utf-8")
    st = run_page(page)
    assert not st.called("sidebar_page_link") and st.called("markdown")


def test_sidebar_skips_missing_page_files(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)               # โปรเจกต์ว่าง ไม่มีไฟล์หน้าเลย
    page = tmp_path / "p.py"
    page.write_text("from src.ui import inject_css\ninject_css()\n", encoding="utf-8")
    st = run_page(page)
    assert not st.called("sidebar_page_link")                        # ไม่ล้ม แค่ไม่แสดงลิงก์


def test_real_streamlit_home_renders_sidebar():
    """รันหน้าแรกกับ Streamlit จริง (AppTest) ตรวจว่าไม่มี exception และเมนูซ้ายถูกวาด"""
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("FitMetrix AI" in m.value and "fm-brand" in m.value for m in at.sidebar.markdown)


def test_no_emoji_in_nav_labels_or_icons():
    import re
    emoji = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]")
    assert not any(emoji.search(label + icon) for _, label, icon in ui.NAV_ITEMS)