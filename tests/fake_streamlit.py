"""
fake_streamlit.py - Streamlit จำลองสำหรับรันสคริปต์หน้าเว็บในเทสต์ (ไม่ต้องมี Streamlit และไม่เปิดเบราว์เซอร์)

ตรวจได้ว่าโค้ดหน้าเว็บ "รันจนจบโดยไม่พัง" (ชื่อตัวแปรผิด ฟังก์ชันผิดพารามิเตอร์ ข้อมูลหาย ฯลฯ)
ตรวจไม่ได้ว่าหน้าตาจริงเป็นอย่างไร หรือ API ของ Streamlit จริงรองรับพารามิเตอร์นั้นหรือไม่
เมธอดที่ไม่ได้นิยามไว้จะ raise AttributeError เพื่อให้จับการเรียกฟังก์ชันที่พิมพ์ผิดได้
"""

from __future__ import annotations

import runpy
import sys
import types
from contextlib import contextmanager


class RerunSignal(Exception):
    pass


class Ctx:
    """context manager ที่ส่งต่อเมธอดทั้งหมดไปที่ st (เช่น col.metric(...) = st.metric(...))"""

    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


class SidebarCtx(Ctx):
    """st.sidebar: page_link ในเมนูซ้ายถูกบันทึกแยกเป็น 'sidebar_page_link' (ไม่ปนกับลิงก์ในหน้า)"""

    def page_link(self, page, **k):
        self._st._rec("sidebar_page_link", page, **k)


class FakeStreamlit:
    def __init__(self, session=None, values=None, submit=False, clicks=()):
        self.sidebar = SidebarCtx(self)
        self.session_state = dict(session or {})
        self.values = dict(values or {})            # ป้ายชื่อ widget -> ค่าที่จะคืน
        self.submit = submit
        self.clicks = set(clicks)
        self.calls: list[tuple] = []                # (ชื่อเมธอด, args, kwargs)
        self.html: list[tuple[str, int]] = []       # ชิ้นส่วนที่ส่งให้ components.html

    # ----- บันทึกการเรียก -----
    def _rec(self, name, *a, **k):
        self.calls.append((name, a, k))

    def texts(self, method=None):
        out = []
        for name, a, _k in self.calls:
            if method and name != method:
                continue
            out += [x for x in a if isinstance(x, str)]
        return out

    def called(self, name):
        return [c for c in self.calls if c[0] == name]

    def _value(self, label, default):
        return self.values.get(label, default)

    # ----- ข้อความ/สถานะ -----
    def set_page_config(self, **k): self._rec("set_page_config", **k)
    def markdown(self, body, **k): self._rec("markdown", body, **k)
    def caption(self, body, **k): self._rec("caption", body, **k)
    def title(self, body, **k): self._rec("title", body)
    def header(self, body, **k): self._rec("header", body)
    def subheader(self, body, **k): self._rec("subheader", body)
    def info(self, body, **k): self._rec("info", body)
    def warning(self, body, **k): self._rec("warning", body)
    def error(self, body, **k): self._rec("error", body)
    def success(self, body, **k): self._rec("success", body)
    def toast(self, body, **k): self._rec("toast", body)
    def divider(self): self._rec("divider")
    def metric(self, label, value, delta=None, **k): self._rec("metric", label, value, delta, **k)
    def table(self, data, **k): self._rec("table", data)
    def dataframe(self, data, **k): self._rec("dataframe", data, **k)
    def page_link(self, page, **k): self._rec("page_link", page, **k)

    def rerun(self):
        self._rec("rerun")
        raise RerunSignal()

    # ----- เลย์เอาต์ -----
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        self._rec("columns", n)
        return [Ctx(self) for _ in range(n)]

    def tabs(self, labels):
        self._rec("tabs", tuple(labels))
        return [Ctx(self) for _ in labels]

    def container(self, **k):
        self._rec("container", **k)
        return Ctx(self)

    def form(self, key, **k):
        self._rec("form", key)
        return Ctx(self)

    def expander(self, label, **k):
        self._rec("expander", label)
        return Ctx(self)

    def spinner(self, text="", **k):
        return Ctx(self)

    # ----- widget -----
    def radio(self, label, options, index=0, format_func=None, horizontal=False, **k):
        options = list(options)
        if format_func:
            for o in options:                       # ต้องแปลงเป็นข้อความได้ทุกตัวเลือก (จับ format_func ที่พัง)
                format_func(o)
        self._rec("radio", label)
        return self._value(label, options[index])

    def selectbox(self, label, options, index=0, format_func=None, **k):
        options = list(options)
        if format_func:
            for o in options:
                format_func(o)
        self._rec("selectbox", label)
        return self._value(label, options[index])

    def select_slider(self, label, options=(), value=None, format_func=None, **k):
        options = list(options)
        if format_func:
            for o in options:
                format_func(o)
        self._rec("select_slider", label)
        return self._value(label, value if value is not None else options[0])

    def slider(self, label, min_value=0, max_value=100, value=None, **k):
        self._rec("slider", label)
        return self._value(label, min_value if value is None else value)

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **k):
        self._rec("number_input", label)
        return self._value(label, min_value if value is None else value)

    def checkbox(self, label, value=False, **k):
        self._rec("checkbox", label)
        return self._value(label, value)

    def button(self, label, **k):
        self._rec("button", label)
        return label in self.clicks

    def form_submit_button(self, label, **k):
        self._rec("form_submit_button", label)
        return self.submit

    def cache_resource(self, func=None, **k):
        return func if func is not None else (lambda f: f)


@contextmanager
def fake_streamlit(**kwargs):
    """ติดตั้ง streamlit จำลองลง sys.modules ชั่วคราว คืนอ็อบเจ็กต์ให้ตรวจผลหลังรัน"""
    st = FakeStreamlit(**kwargs)
    components_v1 = types.ModuleType("streamlit.components.v1")
    components_v1.html = lambda html, height=None, **k: st.html.append((html, height))
    components = types.ModuleType("streamlit.components")
    components.v1 = components_v1
    root = types.ModuleType("streamlit")
    for name in dir(st):
        if not name.startswith("_"):
            setattr(root, name, getattr(st, name))
    root.session_state = st.session_state
    root.components = components
    saved = {k: sys.modules.get(k) for k in ("streamlit", "streamlit.components", "streamlit.components.v1")}
    sys.modules.update({"streamlit": root, "streamlit.components": components, "streamlit.components.v1": components_v1})
    try:
        yield st
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def run_page(path, **kwargs):
    """รันสคริปต์หน้าเว็บหนึ่งรอบ (เหมือน Streamlit รันสคริปต์ใหม่หลังกดปุ่ม) คืน FakeStreamlit"""
    with fake_streamlit(**kwargs) as st:
        try:
            runpy.run_path(str(path), run_name="__main__")
        except RerunSignal:
            pass
        return st