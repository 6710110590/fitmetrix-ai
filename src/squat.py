"""
squat.py - ตรวจจับท่า Squat จากจุดข้อต่อ: นับครั้ง + เตือนท่าผิด

ไม่ผูกกับกล้องหรือ MediaPipe (รับจุดข้อต่อเข้ามา) จึงทดสอบด้วยข้อมูลจำลองได้

เลือกมุมกล้องด้วย cfg.view เพราะแต่ละมุมวัดอะไรได้ไม่เหมือนกัน
  "front" (กล้องโน้ตบุ๊กอยู่ตรงหน้า): นับจาก "สะโพกอยู่เหนือเข่าแค่ไหน" ปรับตามความยาวต้นขาตอนยืน
                                     เตือน: ลงไม่ลึก / เข่าหุบเข้า / สะโพกเอียง
  "side"  (ยืนหันข้างให้กล้อง)        : นับจากมุมเข่า (สะโพก-เข่า-ข้อเท้า)
                                     เตือน: ลงไม่ลึก / ลำตัวโน้มไปข้างหน้ามากไป

ข้อจำกัดที่ต้องรู้:
  - MediaPipe ไม่มีจุดบนกระดูกสันหลัง จึงวัด "หลังงอ/โค้ง" ไม่ได้ วัดได้แค่ลำตัวเอียงจากแนวดิ่ง (มุมสะโพก-ไหล่)
  - เกณฑ์ทุกตัวเป็นค่าตั้งต้นเชิงประมาณ ยังไม่ได้สอบเทียบ ปรับได้ที่ SquatConfig (หรือ --set ตอนรัน)
  - เป็นคำแนะนำเบื้องต้น ไม่ใช่คำแนะนำทางการแพทย์/กายภาพบำบัด
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field, fields
from typing import NamedTuple, Optional, Sequence

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


class Point(NamedTuple):
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class PoseFrame:
    image: Sequence[Point]                      # พิกัดในภาพ normalized 0-1 (x = ความกว้าง, y = ความสูง)
    world: Optional[Sequence[Point]] = None     # พิกัด 3 มิติเป็นเมตร (ถ้ามี ใช้คำนวณมุมเข่าด้านข้างให้แม่นขึ้น)
    aspect: float = 4 / 3                       # กว้าง/สูง ของภาพ ใช้ให้ระยะแนวนอนกับแนวตั้งอยู่หน่วยเดียวกัน


@dataclass
class SquatConfig:
    view: str = "front"                 # "front" | "side"
    min_visibility: float = 0.5
    smoothing: float = 0.5              # EMA alpha (0-1) ยิ่งมากยิ่งไวแต่สั่น
    min_rep_seconds: float = 0.6        # ลงแล้วกลับขึ้นเร็วกว่านี้ถือว่าเป็นสัญญาณรบกวน
    min_issue_frames: int = 4           # ผิดท่าต้องต่อเนื่องอย่างน้อยกี่เฟรมช่วงก้นท่า ถึงจะเตือน
    calib_frames: int = 20              # (front) จำนวนเฟรมยืนตรงที่ใช้วัดความยาวต้นขาอ้างอิง
    # --- ด้านข้าง: มุมเข่า (องศา) ลดลง = ลงลึก ---
    side_up: float = 160.0              # เหยียดตรงพอจะนับว่ายืนแล้ว
    side_start: float = 150.0           # ต่ำกว่านี้ = เริ่มลง
    side_count: float = 120.0           # ลงถึงนี้ถึงจะนับ 1 ครั้ง
    side_good: float = 100.0            # ลงลึกถึงนี้ = ลึกเพียงพอ ไม่เตือน
    # --- ด้านหน้า: สะโพกเหนือเข่า / ความยาวต้นขาตอนยืน (1 = ยืน, 0 = สะโพกเสมอเข่า) ---
    front_up: float = 0.85
    front_start: float = 0.75
    front_count: float = 0.60
    front_good: float = 0.30
    # --- เกณฑ์ตรวจท่า ---
    max_trunk_lean: float = 70.0        # (side) ลำตัวเอียงจากแนวดิ่งเกินนี้ที่ก้นท่า = เตือน
    min_knee_ratio: float = 0.80        # (front) ระยะเข่าสองข้าง / ระยะข้อเท้าสองข้าง ต่ำกว่านี้ = เข่าหุบ
    max_hip_tilt: float = 0.12          # (front) สะโพกสองข้างต่างระดับเกินนี้ (หน่วยความยาวต้นขา) = เตือน

    def thresholds(self) -> tuple[float, float, float, float]:
        """(up, start, count, good) ของมุมกล้องที่เลือก ค่าน้อย = ลงลึก"""
        if self.view == "side":
            return self.side_up, self.side_start, self.side_count, self.side_good
        return self.front_up, self.front_start, self.front_count, self.front_good

    @classmethod
    def from_overrides(cls, **kw) -> "SquatConfig":
        valid = {f.name for f in fields(cls)}
        bad = set(kw) - valid
        if bad:
            raise ValueError(f"ไม่มีพารามิเตอร์: {sorted(bad)} (ที่ใช้ได้: {sorted(valid)})")
        return cls(**kw)


ISSUE_TEXT = {
    "shallow": {"th": "ลงให้ลึกกว่านี้", "en": "Go deeper"},
    "shallow_nocount": {"th": "ลงไม่ลึกพอ ไม่นับครั้งนี้", "en": "Too shallow - not counted"},
    "lean": {"th": "ลำตัวโน้มไปข้างหน้ามากไป ยกอกขึ้น", "en": "Chest up - leaning too far forward"},
    "valgus": {"th": "เข่าหุบเข้า ดันเข่าออกตามปลายเท้า", "en": "Knees caving in - push knees out"},
    "tilt": {"th": "สะโพกเอียง ลงน้ำหนักให้เท่ากันสองข้าง", "en": "Hips uneven - balance both sides"},
}
STATUS_TEXT = {
    "no_pose": {"th": "ไม่พบคนในภาพ", "en": "No person detected"},
    "partial": {"th": "เห็นร่างกายไม่ครบ ถอยออกให้เห็นตั้งแต่ศีรษะถึงเท้า", "en": "Body not fully visible - step back"},
    "calibrating": {"th": "ยืนตรงนิ่งๆ เพื่อตั้งค่าเริ่มต้น", "en": "Stand still to calibrate"},
}


@dataclass
class RepEvent:
    index: Optional[int]        # ลำดับครั้งที่นับ (None = ไม่นับ)
    counted: bool
    min_signal: float           # ค่าต่ำสุด (ลึกสุด) ของครั้งนั้น
    duration: float             # วินาที
    issues: list = field(default_factory=list)


@dataclass
class FrameResult:
    status: str                 # "no_pose" | "partial" | "calibrating" | "ready"
    phase: str                  # "up" | "down"
    reps: int
    signal: Optional[float] = None
    live_issues: list = field(default_factory=list)
    event: Optional[RepEvent] = None
    calibration_progress: float = 0.0


# ----------------------------- เรขาคณิต -------------------------------------
def angle_deg(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> Optional[float]:
    """มุมที่จุด b ระหว่างเส้น b->a กับ b->c (องศา) รับจุด 2 หรือ 3 มิติ"""
    ba = [p - q for p, q in zip(a, b)]
    bc = [p - q for p, q in zip(c, b)]
    na, nc = math.sqrt(sum(v * v for v in ba)), math.sqrt(sum(v * v for v in bc))
    if na < 1e-9 or nc < 1e-9:
        return None
    cos = sum(p * q for p, q in zip(ba, bc)) / (na * nc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _dist2(a: Point, b: Point, aspect: float) -> float:
    return math.hypot((a.x - b.x) * aspect, a.y - b.y)


class _Measure(NamedTuple):
    knee_angle: Optional[float] = None      # side
    trunk_lean: Optional[float] = None      # side
    thigh_len: Optional[float] = None       # front
    hip_over_knee: Optional[float] = None   # front (หน่วยภาพ ยังไม่หารด้วยความยาวต้นขาอ้างอิง)
    knee_ratio: Optional[float] = None      # front
    hip_tilt: Optional[float] = None        # front (หน่วยภาพ)


class SquatAnalyzer:
    def __init__(self, config: Optional[SquatConfig] = None):
        self.cfg = config or SquatConfig()
        if self.cfg.view not in ("front", "side"):
            raise ValueError("view ต้องเป็น 'front' หรือ 'side'")
        self.reset()

    # ----- สถานะ -----
    def reset(self) -> None:
        self.reps = 0
        self.events: list[RepEvent] = []
        self.recalibrate()

    def recalibrate(self) -> None:
        self.phase = "up"
        self._baseline: Optional[float] = None
        self._calib: list[float] = []
        self._ema: Optional[float] = None
        self._min_signal: Optional[float] = None
        self._t_start = 0.0
        self._issue_frames: dict[str, int] = {}

    @property
    def signal_label(self) -> str:
        return "knee angle" if self.cfg.view == "side" else "hip/knee"

    # ----- วัดจากจุดข้อต่อ -----
    def _measure(self, f: PoseFrame) -> Optional[_Measure]:
        thr, img = self.cfg.min_visibility, f.image
        if self.cfg.view == "front":
            ids = (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)
            if any(img[i].visibility < thr for i in ids):
                return None
            hl, hr, kl, kr, al, ar = (img[i] for i in ids)
            thigh = (_dist2(hl, kl, f.aspect) + _dist2(hr, kr, f.aspect)) / 2
            dy = (kl.y + kr.y) / 2 - (hl.y + hr.y) / 2
            ankle_dx = abs(al.x - ar.x) * f.aspect
            knee_dx = abs(kl.x - kr.x) * f.aspect
            return _Measure(thigh_len=thigh, hip_over_knee=dy,
                            knee_ratio=knee_dx / ankle_dx if ankle_dx > 1e-6 else None,
                            hip_tilt=abs(hl.y - hr.y))

        # side: ใช้ขาข้างที่เห็นชัดกว่า
        best = None
        for hip, knee, ankle, sh in ((L_HIP, L_KNEE, L_ANKLE, L_SHOULDER), (R_HIP, R_KNEE, R_ANKLE, R_SHOULDER)):
            vis = [img[i].visibility for i in (hip, knee, ankle)]
            if min(vis) >= thr and (best is None or sum(vis) > best[0]):
                best = (sum(vis), hip, knee, ankle, sh)
        if best is None:
            return None
        _, hip, knee, ankle, sh = best
        if f.world is not None and all(f.world[i].visibility >= thr for i in (hip, knee, ankle)):
            pts = [(f.world[i].x, f.world[i].y, f.world[i].z) for i in (hip, knee, ankle)]
        else:
            pts = [(img[i].x * f.aspect, img[i].y) for i in (hip, knee, ankle)]
        angle = angle_deg(*pts)
        if angle is None:
            return None
        lean = None
        if img[sh].visibility >= thr:
            lean = math.degrees(math.atan2(abs((img[sh].x - img[hip].x) * f.aspect), abs(img[sh].y - img[hip].y)))
        return _Measure(knee_angle=angle, trunk_lean=lean)

    # ----- เฟรมต่อเฟรม -----
    def _lost(self, status: str) -> FrameResult:
        self.phase = "up"                   # ทิ้งครั้งที่กำลังทำอยู่ (มองไม่เห็น วัดไม่ได้)
        self._ema = None
        self._min_signal = None
        return FrameResult(status=status, phase=self.phase, reps=self.reps)

    def update(self, frame: Optional[PoseFrame], t: float) -> FrameResult:
        cfg = self.cfg
        if frame is None:
            return self._lost("no_pose")
        m = self._measure(frame)
        if m is None:
            return self._lost("partial")

        if cfg.view == "front":
            if self._baseline is None:
                # ตั้งค่าอ้างอิง: เก็บความยาวต้นขาตอนยืนตรง (ต้นขาแนวตั้ง: สะโพกอยู่เหนือเข่าเกือบเต็มความยาว)
                if m.thigh_len and m.hip_over_knee / m.thigh_len >= 0.9:
                    self._calib.append(m.thigh_len)
                if len(self._calib) >= cfg.calib_frames:
                    self._baseline = statistics.median(self._calib)
                else:
                    return FrameResult(status="calibrating", phase=self.phase, reps=self.reps,
                                       calibration_progress=len(self._calib) / cfg.calib_frames)
            raw = m.hip_over_knee / self._baseline
        else:
            raw = m.knee_angle

        self._ema = raw if self._ema is None else cfg.smoothing * raw + (1 - cfg.smoothing) * self._ema
        s = self._ema
        up, start, count, _good = cfg.thresholds()
        live: list[str] = []
        event: Optional[RepEvent] = None

        if self.phase == "up":
            if s < start:
                self.phase, self._min_signal, self._t_start, self._issue_frames = "down", s, t, {}
        else:
            self._min_signal = min(self._min_signal, s)
            if s <= count:                                   # ตรวจท่าเมื่อลงมาถึงช่วงก้นท่า
                live = self._form_issues(m)
                for code in live:
                    self._issue_frames[code] = self._issue_frames.get(code, 0) + 1
            if s >= up:
                event = self._finish(t)
                self.phase = "up"

        return FrameResult(status="ready", phase=self.phase, reps=self.reps, signal=s,
                           live_issues=live, event=event)

    def _form_issues(self, m: _Measure) -> list[str]:
        cfg, issues = self.cfg, []
        if cfg.view == "side":
            if m.trunk_lean is not None and m.trunk_lean > cfg.max_trunk_lean:
                issues.append("lean")
        else:
            if m.knee_ratio is not None and m.knee_ratio < cfg.min_knee_ratio:
                issues.append("valgus")
            if m.hip_tilt is not None and m.hip_tilt / self._baseline > cfg.max_hip_tilt:
                issues.append("tilt")
        return issues

    def _finish(self, t: float) -> Optional[RepEvent]:
        duration = t - self._t_start
        if duration < self.cfg.min_rep_seconds:
            return None                                       # เร็วเกินไป = สัญญาณรบกวน ไม่นับและไม่เตือน
        _up, _start, count, good = self.cfg.thresholds()
        if self._min_signal <= count:
            self.reps += 1
            issues = [c for c, n in self._issue_frames.items() if n >= self.cfg.min_issue_frames]
            if self._min_signal > good:
                issues.insert(0, "shallow")
            ev = RepEvent(index=self.reps, counted=True, min_signal=round(self._min_signal, 3),
                          duration=round(duration, 2), issues=issues)
        else:
            ev = RepEvent(index=None, counted=False, min_signal=round(self._min_signal, 3),
                          duration=round(duration, 2), issues=["shallow_nocount"])
        self.events.append(ev)
        return ev

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for ev in self.events:
            for code in ev.issues:
                counts[code] = counts.get(code, 0) + 1
        return {
            "view": self.cfg.view,
            "reps": self.reps,
            "attempts": len(self.events),
            "issue_counts": counts,
            "events": [asdict(e) for e in self.events],
            "config": asdict(self.cfg),
        }