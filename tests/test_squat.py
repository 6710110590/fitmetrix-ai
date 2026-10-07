import math

import pytest

from src.squat import (
    FrameResult,
    Point,
    PoseFrame,
    SquatAnalyzer,
    SquatConfig,
    angle_deg,
)

FPS = 30.0


# ---------- ตัวสร้างท่าจำลอง ----------
def _blank(vis=0.99):
    return [Point(0.5, 0.5, 0.0, vis) for _ in range(33)]


def side_frame(knee_angle, lean=30.0, vis=0.99, world=False):
    """ท่ายืนหันข้าง: ข้อเท้า (0.5, 0.9) ต้นขา/หน้าแข้งยาว 0.22 มุมเข่า = knee_angle องศา"""
    ax, ay, l1, l2, lt = 0.5, 0.9, 0.22, 0.22, 0.30
    alpha = math.radians(min((180 - knee_angle) * 0.3, 30))             # หน้าแข้งเอียงไปข้างหน้า
    kx, ky = ax + l2 * math.sin(alpha), ay - l2 * math.cos(alpha)
    ux, uy = -math.sin(alpha), math.cos(alpha)                           # ทิศเข่า -> ข้อเท้า
    th = math.radians(knee_angle)
    vx, vy = ux * math.cos(th) - uy * math.sin(th), ux * math.sin(th) + uy * math.cos(th)
    hx, hy = kx + l1 * vx, ky + l1 * vy
    lr = math.radians(lean)
    sx, sy = hx + lt * math.sin(lr), hy - lt * math.cos(lr)
    pts = _blank(vis)
    for i in (11, 12):
        pts[i] = Point(sx, sy, 0.0, vis)
    for i in (23, 24):
        pts[i] = Point(hx, hy, 0.0, vis)
    for i in (25, 26):
        pts[i] = Point(kx, ky, 0.0, vis)
    for i in (27, 28):
        pts[i] = Point(ax, ay, 0.0, vis)
    w = None
    if world:                                                            # พิกัดเมตรที่สอดคล้องกัน (z = 0)
        w = [Point(p.x, p.y, 0.0, p.visibility) for p in pts]
    return PoseFrame(image=pts, world=w, aspect=1.0)


def front_frame(r, knee_x=(0.43, 0.57), ankle_x=(0.45, 0.55), hip_tilt=0.0, vis=0.99):
    """ท่ายืนหันหน้า: r = สะโพกเหนือเข่า / ความยาวต้นขา (1 = ยืน, 0 = สะโพกเสมอเข่า)"""
    knee_y, big_l = 0.70, 0.20
    hip_y = knee_y - r * big_l
    pts = _blank(vis)
    pts[23] = Point(0.46, hip_y - hip_tilt * big_l / 2, 0.0, vis)
    pts[24] = Point(0.54, hip_y + hip_tilt * big_l / 2, 0.0, vis)
    pts[25], pts[26] = Point(knee_x[0], knee_y, 0.0, vis), Point(knee_x[1], knee_y, 0.0, vis)
    pts[27], pts[28] = Point(ankle_x[0], 0.90, 0.0, vis), Point(ankle_x[1], 0.90, 0.0, vis)
    return PoseFrame(image=pts, aspect=1.0)


def profile(top, bottom, frames=60):
    """ค่าสัญญาณหนึ่งครั้ง: ลงจาก top ถึง bottom แล้วกลับขึ้น (cosine) ใช้ราว 2 วินาที"""
    return [top - (top - bottom) * (0.5 - 0.5 * math.cos(2 * math.pi * i / frames)) for i in range(frames + 1)]


def run(analyzer, frames, t0=0.0):
    results, t = [], t0
    for f in frames:
        results.append(analyzer.update(f, t))
        t += 1 / FPS
    return results, t


def side_session(depths, top=172.0, rest=15, **kw):
    seq = [side_frame(top, **kw) for _ in range(30)]
    for d in depths:
        seq += [side_frame(a, **kw) for a in profile(top, d)]
        seq += [side_frame(top, **kw) for _ in range(rest)]
    return seq


def front_session(depths, top=1.0, rest=15, **kw):
    seq = [front_frame(top) for _ in range(30)]
    for d in depths:
        seq += [front_frame(r, **kw) for r in profile(top, d)]
        seq += [front_frame(top) for _ in range(rest)]
    return seq


def events(results):
    return [r.event for r in results if r.event]


# ---------- เรขาคณิต ----------
def test_angle_deg():
    assert angle_deg((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0)
    assert angle_deg((1, 0), (0, 0), (-1, 0)) == pytest.approx(180.0)
    assert angle_deg((1, 0, 0), (0, 0, 0), (0, 0, 1)) == pytest.approx(90.0)
    assert angle_deg((0, 0), (0, 0), (1, 0)) is None


def test_synthetic_side_pose_has_requested_knee_angle():
    a = SquatAnalyzer(SquatConfig(view="side"))
    m = a._measure(side_frame(100.0))
    assert m.knee_angle == pytest.approx(100.0, abs=0.5)


def test_config_overrides_validate_names():
    assert SquatConfig.from_overrides(view="side", side_count=130).side_count == 130
    with pytest.raises(ValueError):
        SquatConfig.from_overrides(not_a_param=1)
    with pytest.raises(ValueError):
        SquatAnalyzer(SquatConfig(view="top"))


# ---------- ด้านข้าง ----------
def test_side_counts_five_good_reps():
    a = SquatAnalyzer(SquatConfig(view="side"))
    res, _ = run(a, side_session([85] * 5))
    evs = events(res)
    assert a.reps == 5 and [e.index for e in evs] == [1, 2, 3, 4, 5]
    assert all(e.counted and not e.issues for e in evs)


def test_side_shallow_rep_counts_but_warns():
    a = SquatAnalyzer(SquatConfig(view="side"))
    res, _ = run(a, side_session([110]))                 # ถึง 120 (นับ) แต่ไม่ถึง 100 (ลึกพอ)
    ev = events(res)[0]
    assert ev.counted and ev.issues == ["shallow"]


def test_side_too_shallow_not_counted():
    a = SquatAnalyzer(SquatConfig(view="side"))
    res, _ = run(a, side_session([135]))                 # เลย 150 (เริ่มลง) แต่ไม่ถึง 120
    ev = events(res)[0]
    assert a.reps == 0 and not ev.counted and ev.issues == ["shallow_nocount"]


def test_side_trunk_lean_warning_only_when_leaning():
    a = SquatAnalyzer(SquatConfig(view="side"))
    res, _ = run(a, side_session([85], lean=78))
    assert "lean" in events(res)[0].issues
    b = SquatAnalyzer(SquatConfig(view="side"))
    res, _ = run(b, side_session([85], lean=40))
    assert "lean" not in events(res)[0].issues


def test_side_with_world_landmarks_gives_same_count():
    a = SquatAnalyzer(SquatConfig(view="side"))
    run(a, [side_frame(x, world=True) for x in side_session_angles([85] * 3)])
    assert a.reps == 3


def side_session_angles(depths, top=172.0, rest=15):
    out = [top] * 30
    for d in depths:
        out += profile(top, d) + [top] * rest
    return out


def test_side_picks_the_more_visible_leg():
    a = SquatAnalyzer(SquatConfig(view="side"))
    f = side_frame(100.0)
    pts = list(f.image)
    for i in (23, 25, 27):                               # ขาซ้ายถูกบัง
        pts[i] = pts[i]._replace(visibility=0.1)
    m = a._measure(PoseFrame(image=pts, aspect=1.0))
    assert m is not None and m.knee_angle == pytest.approx(100.0, abs=0.5)


# ---------- ด้านหน้า ----------
def test_front_calibrates_then_counts_five_reps():
    a = SquatAnalyzer(SquatConfig(view="front"))
    res, _ = run(a, front_session([0.1] * 5))
    assert res[0].status == "calibrating" and res[0].calibration_progress < 1
    assert any(r.status == "ready" for r in res)
    assert a.reps == 5 and all(e.counted and not e.issues for e in events(res))


def test_front_does_not_calibrate_while_squatting():
    a = SquatAnalyzer(SquatConfig(view="front"))
    res, _ = run(a, [front_frame(0.3) for _ in range(40)])
    assert all(r.status == "calibrating" for r in res) and a._baseline is None


def test_front_shallow_not_counted():
    a = SquatAnalyzer(SquatConfig(view="front"))
    res, _ = run(a, front_session([0.65]))
    ev = events(res)[0]
    assert a.reps == 0 and ev.issues == ["shallow_nocount"]


def test_front_knee_valgus_warning():
    a = SquatAnalyzer(SquatConfig(view="front"))
    res, _ = run(a, front_session([0.1], knee_x=(0.47, 0.53)))
    assert "valgus" in events(res)[0].issues
    b = SquatAnalyzer(SquatConfig(view="front"))
    res, _ = run(b, front_session([0.1]))
    assert "valgus" not in events(res)[0].issues


def test_front_hip_tilt_warning():
    a = SquatAnalyzer(SquatConfig(view="front"))
    res, _ = run(a, front_session([0.1], hip_tilt=0.25))
    assert "tilt" in events(res)[0].issues


def test_live_issues_reported_at_the_bottom():
    a = SquatAnalyzer(SquatConfig(view="front"))
    res, _ = run(a, front_session([0.1], knee_x=(0.47, 0.53)))
    assert any("valgus" in r.live_issues for r in res)
    assert not any(r.live_issues for r in res[:30])


# ---------- ความทนทานต่อสัญญาณรบกวน ----------
def test_noise_while_standing_never_counts():
    a = SquatAnalyzer(SquatConfig(view="front"))
    seq = [front_frame(1.0)] * 30 + [front_frame(1.0 + 0.04 * math.sin(i)) for i in range(120)]
    res, _ = run(a, seq)
    assert a.reps == 0 and not events(res)


def test_too_fast_dip_is_ignored():
    a = SquatAnalyzer(SquatConfig(view="front"))
    seq = [front_frame(1.0)] * 30 + [front_frame(0.1)] * 3 + [front_frame(1.0)] * 20
    res, _ = run(a, seq)
    assert a.reps == 0 and not events(res)


def test_jitter_around_threshold_counts_once():
    a = SquatAnalyzer(SquatConfig(view="front"))
    bottom = [front_frame(0.30 + 0.25 * (0.5 + 0.5 * math.sin(i))) for i in range(60)]   # แกว่ง 0.30-0.55 ตลอดก้นท่า
    seq = [front_frame(1.0)] * 30 + [front_frame(r) for r in profile(1.0, 0.45, 20)] + bottom + \
          [front_frame(r) for r in reversed(profile(1.0, 0.45, 20)[:11])] + [front_frame(1.0)] * 20
    res, _ = run(a, seq)
    assert a.reps == 1


def test_losing_the_body_aborts_the_rep():
    a = SquatAnalyzer(SquatConfig(view="front"))
    seq = [front_frame(1.0)] * 30 + [front_frame(r) for r in profile(1.0, 0.1)[:25]]
    seq += [front_frame(0.5, vis=0.1)] * 5                 # มองไม่เห็นกลางครั้ง
    seq += [front_frame(r) for r in profile(1.0, 0.1)[40:]] + [front_frame(1.0)] * 10
    res, _ = run(a, seq)
    assert any(r.status == "partial" for r in res) and a.reps == 0


def test_no_pose_status():
    a = SquatAnalyzer()
    assert a.update(None, 0.0).status == "no_pose"


def test_reset_and_recalibrate():
    a = SquatAnalyzer(SquatConfig(view="front"))
    run(a, front_session([0.1] * 2))
    assert a.reps == 2
    a.recalibrate()
    assert a.reps == 2 and a._baseline is None
    a.reset()
    assert a.reps == 0 and not a.events


def test_summary_structure():
    a = SquatAnalyzer(SquatConfig(view="side"))
    run(a, side_session([85, 110, 135]))
    s = a.summary()
    assert s["reps"] == 2 and s["attempts"] == 3 and s["view"] == "side"
    assert s["issue_counts"] == {"shallow": 1, "shallow_nocount": 1}
    assert len(s["events"]) == 3 and "config" in s