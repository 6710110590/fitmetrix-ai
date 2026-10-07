import json

import numpy as np
import pytest

import live_squat
from live_squat import (
    draw_hud,
    draw_pose,
    find_thai_font,
    hud_lines,
    parse_overrides,
    run_live,
    save_log,
)
from src.squat import FrameResult, RepEvent, SquatAnalyzer, SquatConfig

pytest.importorskip("cv2")
from tests.test_squat import FPS, front_frame, front_session, profile, side_frame, side_session  # noqa: E402

H, W = 120, 160


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 1 / FPS
        return self.t


class FakeCap:
    def __init__(self, n):
        self.n, self.i = n, 0

    def read(self):
        self.i += 1
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        return (True, frame) if self.i <= self.n else (False, None)


class ScriptedDetector:
    """คืนท่าจำลองตามลำดับทีละเฟรม"""

    def __init__(self, poses):
        self.poses, self.i, self.timestamps = poses, 0, []

    def __call__(self, frame_rgb, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        pose = self.poses[min(self.i, len(self.poses) - 1)]
        self.i += 1
        return pose


def run_session(poses, view="front", **kw):
    frames, shown = [], []

    def show(frame):
        shown.append(frame.copy())
        return -1

    analyzer = SquatAnalyzer(SquatConfig(view=view))
    det = ScriptedDetector(poses)
    summary = run_live(FakeCap(len(poses)), det, analyzer, show, clock=FakeClock(), max_frames=len(poses), **kw)
    return summary, shown, det


def test_run_live_counts_reps_front_view():
    summary, shown, det = run_session(front_session([0.1] * 3))
    assert summary["reps"] == 3 and len(shown) > 0
    assert det.timestamps == sorted(det.timestamps) and len(set(det.timestamps)) > 1


def test_run_live_counts_reps_side_view():
    summary, _, _ = run_session(side_session([85] * 2), view="side")
    assert summary["reps"] == 2


def test_beeper_called_only_when_there_are_issues():
    beeps = []
    run_session(front_session([0.1] * 2), beeper=lambda: beeps.append(1))
    assert beeps == []                                           # ท่าถูกต้อง ไม่ส่งเสียง
    run_session(front_session([0.1] * 2, knee_x=(0.47, 0.53)), beeper=lambda: beeps.append(1))
    assert len(beeps) == 2                                       # เข่าหุบทั้งสองครั้ง


def test_hud_is_drawn_on_frames():
    _, shown, _ = run_session(front_session([0.1]))
    assert any(f.sum() > 0 for f in shown)                       # ภาพจากกล้องจำลองดำสนิท ถ้ามีสีแปลว่าวาด HUD แล้ว


def test_keys_quit_reset_recalibrate():
    keys = iter([-1, -1, ord("c"), ord("r"), ord("q")])
    analyzer = SquatAnalyzer(SquatConfig(view="front"))
    analyzer.reps = 4
    seen = []
    summary = run_live(FakeCap(50), ScriptedDetector([front_frame(1.0)]), analyzer,
                       lambda f: seen.append(1) or next(keys), clock=FakeClock())
    assert len(seen) == 5 and summary["reps"] == 0               # r รีเซ็ตจำนวนครั้ง q จบลูป


def test_camera_failure_raises_after_many_misses():
    class DeadCap:
        def read(self):
            return False, None

    with pytest.raises(live_squat.CameraError):
        run_live(DeadCap(), ScriptedDetector([None]), SquatAnalyzer(), lambda f: -1, clock=FakeClock())


def test_hud_lines_content_and_languages():
    ready = FrameResult(status="ready", phase="down", reps=3, signal=0.4, live_issues=["valgus"])
    ev = RepEvent(index=3, counted=True, min_signal=0.1, duration=2.0, issues=["shallow"])
    th = hud_lines(ready, ev, 1.0, "th")
    en = hud_lines(ready, ev, 1.0, "en")
    assert len(th) == len(en) == 2 and th[0][1] == live_squat.RED and en[0][0].startswith("Knees")
    assert hud_lines(ready, ev, 5.0, "en")[0][0].startswith("Knees") and len(hud_lines(ready, ev, 5.0, "en")) == 1   # เลย 3 วินาทีเลิกแสดงผลครั้งก่อน
    good = hud_lines(FrameResult(status="ready", phase="up", reps=1), RepEvent(1, True, 0.1, 2.0, []), 0.5, "en")
    assert good[0][1] == live_squat.GREEN
    calib = hud_lines(FrameResult(status="calibrating", phase="up", reps=0, calibration_progress=0.5), None, 0, "en")
    assert "50%" in calib[0][0]


def test_draw_hud_english_and_thai_paths_do_not_crash():
    analyzer = SquatAnalyzer(SquatConfig(view="front"))
    result = FrameResult(status="ready", phase="down", reps=2, signal=0.33, live_issues=["valgus", "tilt"])
    ev = RepEvent(2, True, 0.1, 2.0, ["shallow"])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(frame, result, analyzer, ev, 1.0, font_path=None)
    assert frame.sum() > 0
    font = find_thai_font()
    if font:
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_hud(frame2, result, analyzer, ev, 1.0, font_path=font)
        assert frame2.sum() > 0 and frame2.shape == (480, 640, 3)


def test_draw_pose_handles_none_and_low_visibility():
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    draw_pose(frame, None)
    assert frame.sum() == 0
    draw_pose(frame, front_frame(1.0, vis=0.1))
    assert frame.sum() == 0
    draw_pose(frame, front_frame(1.0))
    assert frame.sum() > 0


def test_save_log_writes_privacy_safe_summary(tmp_path):
    summary, _, _ = run_session(front_session([0.1]))
    path = save_log(summary, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["reps"] == 1 and "created_at" in data and "config" in data
    assert set(data) >= {"view", "reps", "attempts", "issue_counts", "events"}


def test_parse_overrides():
    assert parse_overrides(["side_count=130", "front_good=0.25", "min_issue_frames=6"]) == \
        {"side_count": 130, "front_good": 0.25, "min_issue_frames": 6}
    with pytest.raises(SystemExit):
        parse_overrides(["bad"])


def test_font_without_thai_glyphs_is_rejected():
    """ฟอนต์ที่ไม่มีตัวอักษรไทยจะวาดเป็นกล่องสี่เหลี่ยม ต้องไม่ถูกเลือกมาใช้"""
    import os

    from live_squat import font_supports_thai

    no_thai = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(no_thai):
        assert not font_supports_thai(no_thai)
        assert find_thai_font([no_thai]) is None
    assert not font_supports_thai("/path/that/does/not/exist.ttf")
    assert find_thai_font([]) is None


def test_thai_font_found_is_verified_to_support_thai():
    from live_squat import font_supports_thai

    font = find_thai_font()
    assert font is None or font_supports_thai(font)