"""
live_squat.py - นับ Squat และเตือนท่าผิดแบบเรียลไทม์จากเว็บแคม (รันบนเครื่องที่ต่อกล้อง)

    python live_squat.py --view front        # กล้องโน้ตบุ๊กอยู่ตรงหน้า (ค่าเริ่มต้น)
    python live_squat.py --view side         # ยืนหันข้างให้กล้อง (วัดความลึก/ลำตัวเอียงได้ดีกว่า)

ปุ่ม: Q หรือ ESC = จบ | R = เริ่มนับใหม่ | C = ตั้งค่าเริ่มต้นใหม่ (ยืนนิ่งๆ ตอนกด)
ปรับเกณฑ์ตอนรัน:  --set side_count=130 --set front_good=0.25   (ดูชื่อทั้งหมดใน SquatConfig)

ความเป็นส่วนตัว: ประมวลผลบนเครื่อง ไม่บันทึกภาพ/วิดีโอ เก็บเฉพาะสรุปตัวเลขลง logs/ (จำนวนครั้ง มุมต่ำสุด ข้อเตือน)
ใช้ได้เฉพาะเมื่อรันบนเครื่องที่มีกล้อง (ใช้ไม่ได้เมื่อ deploy ออนไลน์)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.squat import (  # noqa: E402
    ISSUE_TEXT, STATUS_TEXT, FrameResult, Point, PoseFrame, RepEvent, SquatAnalyzer, SquatConfig,
)

MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
LOG_DIR = ROOT / "logs"
WINDOW = "FitMetrix - Squat"

# เส้นเชื่อมจุดข้อต่อที่วาดบนจอ (ไหล่ แขน ลำตัว สะโพก ขา)
CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
               (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

THAI_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\LeelawUI.ttf", r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\leelawad.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/usr/share/fonts/opentype/tlwg/Loma.otf", "/usr/share/fonts/truetype/tlwg/Loma.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
]

GREEN, YELLOW, RED, WHITE = (80, 200, 80), (0, 215, 255), (60, 60, 255), (255, 255, 255)   # BGR


class CameraError(RuntimeError):
    pass


# ----------------------------- กล้อง / ตัวตรวจจับ ----------------------------
def open_camera(index: int = 0, width: int = 640, height: int = 480):
    import cv2

    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
        cap.release()
    raise CameraError(f"เปิดกล้องหมายเลข {index} ไม่ได้ ตรวจสอบว่าไม่มีโปรแกรมอื่นใช้กล้องอยู่ แล้วลอง --camera 1")


class LiveDetector:
    """MediaPipe PoseLandmarker โหมดวิดีโอ (ติดตามต่อเนื่อง เร็วกว่าตรวจทีละภาพ) ไม่ใช้เงาร่างกาย"""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise SystemExit(f"ไม่พบไฟล์โมเดล {model_path}\nรัน: python setup_scanner.py เพื่อดาวน์โหลด")
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def __call__(self, frame_rgb: np.ndarray, timestamp_ms: int) -> Optional[PoseFrame]:
        ts = max(timestamp_ms, self._last_ts + 1)             # VIDEO mode ต้องการเวลาที่เพิ่มขึ้นเสมอ
        self._last_ts = ts
        h, w = frame_rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(mp_image, ts)
        if not result.pose_landmarks:
            return None

        def to_points(lms):
            return [Point(p.x, p.y, p.z, p.visibility if p.visibility is not None else 0.0) for p in lms]

        world = to_points(result.pose_world_landmarks[0]) if result.pose_world_landmarks else None
        return PoseFrame(image=to_points(result.pose_landmarks[0]), world=world, aspect=w / h)

    def close(self) -> None:
        self._landmarker.close()


# ----------------------------- วาดบนจอ -------------------------------------
def font_supports_thai(path: str) -> bool:
    """ฟอนต์ที่ไม่มีตัวอักษรไทยจะวาดเป็นกล่องสี่เหลี่ยม เช็คโดยเทียบกับตัวอักษรที่ไม่มีอยู่จริง"""
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(path, 32)
        return bytes(font.getmask("ก")) != bytes(font.getmask("\uffff"))
    except Exception:  # noqa: BLE001
        return False


def find_thai_font(candidates: Optional[list] = None) -> Optional[str]:
    """ฟอนต์แรกที่มีอยู่และรองรับภาษาไทยจริง ไม่เจอคืน None (HUD จะใช้ภาษาอังกฤษแทน)"""
    for p in candidates if candidates is not None else THAI_FONT_CANDIDATES:
        if Path(p).exists() and font_supports_thai(p):
            return p
    return None


def draw_pose(frame: np.ndarray, pose: Optional[PoseFrame], min_visibility: float = 0.5) -> None:
    import cv2

    if pose is None:
        return
    h, w = frame.shape[:2]
    pts = pose.image
    for a, b in CONNECTIONS:
        if pts[a].visibility >= min_visibility and pts[b].visibility >= min_visibility:
            cv2.line(frame, (int(pts[a].x * w), int(pts[a].y * h)), (int(pts[b].x * w), int(pts[b].y * h)), (255, 200, 0), 2, cv2.LINE_AA)
    for i in {i for c in CONNECTIONS for i in c}:
        if pts[i].visibility >= min_visibility:
            cv2.circle(frame, (int(pts[i].x * w), int(pts[i].y * h)), 4, (0, 140, 255), -1, cv2.LINE_AA)


def border_style(result: FrameResult, last_event: Optional[RepEvent], event_age: float) -> tuple[tuple, int]:
    """
    สีและความหนากรอบรอบภาพ (BGR, พิกเซลที่ภาพสูง 480)
      เหลือง = ยังไม่พร้อม (ไม่เห็นตัว/เห็นไม่ครบ/กำลังตั้งค่า) หรือครั้งที่เพิ่งจบนับแต่มีข้อเตือน
      แดง    = ผิดท่าอยู่ตอนนี้ หรือครั้งที่เพิ่งจบไม่นับ (ลงไม่ลึกพอ)
      เขียว  = ท่าปกติ (บางๆ) หรือครั้งที่เพิ่งจบท่าดี (หนาขึ้น 1.5 วินาที)
    """
    if result.status != "ready":
        return YELLOW, 6
    if result.live_issues:
        return RED, 14
    if last_event is not None and event_age < 1.5:
        if "shallow_nocount" in last_event.issues:
            return RED, 14
        return (YELLOW if last_event.issues else GREEN), 14
    return GREEN, 6


def draw_border(frame: np.ndarray, color: tuple, thickness: int) -> None:
    """วาดกรอบรอบภาพทั้งสี่ด้าน (ความหนาปรับตามความสูงภาพ) ใช้การเติมสีตรงๆ ให้ความหนาแน่นอน"""
    h, w = frame.shape[:2]
    t = max(2, int(thickness * h / 480))
    frame[:t, :] = color
    frame[h - t:, :] = color
    frame[:, :t] = color
    frame[:, w - t:] = color


def hud_lines(result: FrameResult, last_event: Optional[RepEvent], event_age: float, lang: str) -> list[tuple[str, tuple]]:
    """ข้อความที่จะแสดงมุมล่างจอ [(ข้อความ, สี BGR)] ลำดับสำคัญก่อน"""
    lines: list[tuple[str, tuple]] = []
    if result.status != "ready":
        text = STATUS_TEXT[result.status][lang]
        if result.status == "calibrating":
            text += f" ({int(result.calibration_progress * 100)}%)"
        lines.append((text, YELLOW))
    for code in result.live_issues:
        lines.append((ISSUE_TEXT[code][lang], RED))
    if last_event is not None and event_age < 3.0 and result.status == "ready":
        if not last_event.issues:
            lines.append(("ดีมาก ท่าถูกต้อง" if lang == "th" else "Good rep!", GREEN))
        for code in last_event.issues:
            lines.append((ISSUE_TEXT[code][lang], YELLOW))
    return lines


def draw_hud(frame: np.ndarray, result: FrameResult, analyzer: SquatAnalyzer, last_event: Optional[RepEvent],
             event_age: float, font_path: Optional[str] = None) -> None:
    """วาดจำนวนครั้ง ค่าสัญญาณ และข้อความเตือน (ภาษาไทยถ้ามีฟอนต์ ไม่มีก็ใช้ภาษาอังกฤษ)"""
    import cv2

    h, w = frame.shape[:2]
    s = h / 480
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(text, xy, scale, color, thick):
        cv2.putText(frame, text, xy, font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(frame, text, xy, font, scale, color, thick, cv2.LINE_AA)

    put(f"REPS {result.reps}", (int(14 * s), int(52 * s)), 1.5 * s, WHITE, max(2, int(3 * s)))
    if result.signal is not None:
        up, start, count, good = analyzer.cfg.thresholds()
        put(f"{analyzer.signal_label}: {result.signal:.2f}  (count<={count:g} good<={good:g})", (int(14 * s), int(84 * s)), 0.5 * s, WHITE, 1)
    put("Q quit | R reset | C recalibrate", (int(14 * s), h - int(10 * s)), 0.45 * s, (200, 200, 200), 1)

    lang = "th" if font_path else "en"
    lines = hud_lines(result, last_event, event_age, lang)
    if not lines:
        return
    if font_path:
        from PIL import Image, ImageDraw, ImageFont

        size = max(16, int(26 * s))
        pil_font = ImageFont.truetype(font_path, size)
        img = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
        draw = ImageDraw.Draw(img)
        y = h - int(36 * s) - size * len(lines) * 1.3
        for text, (b, g, r) in lines:
            draw.text((int(14 * s), y), text, font=pil_font, fill=(r, g, b), stroke_width=2, stroke_fill=(0, 0, 0))
            y += size * 1.3
        frame[:] = np.asarray(img)[:, :, ::-1]
    else:
        y = h - int(36 * s) - int(30 * s) * (len(lines) - 1)
        for text, color in lines:
            put(text, (int(14 * s), y), 0.8 * s, color, max(1, int(2 * s)))
            y += int(30 * s)


def make_beeper() -> Callable[[], None]:
    """เสียงเตือนเมื่อมีข้อผิดท่า (เฉพาะ Windows ระบบอื่นไม่มีเสียง) เล่นแบบไม่บล็อกลูป"""
    try:
        import winsound
    except ImportError:
        return lambda: None
    return lambda: winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)


# ----------------------------- ลูปหลัก --------------------------------------
def run_live(cap, detect: Callable, analyzer: SquatAnalyzer, show: Callable[[np.ndarray], int],
             beeper: Optional[Callable[[], None]] = None, clock: Callable[[], float] = time.monotonic,
             max_frames: Optional[int] = None, font_path: Optional[str] = None) -> dict:
    """
    cap.read() -> (ok, frame BGR) | detect(frame_rgb, timestamp_ms) -> PoseFrame | None
    show(frame) แสดงภาพและคืนรหัสปุ่มที่กด (-1 = ไม่มี) | คืน summary ของ SquatAnalyzer
    แยกออกมารับ cap/detect/show เป็นพารามิเตอร์ เพื่อทดสอบด้วยกล้องจำลองได้
    """
    last_event: Optional[RepEvent] = None
    last_event_t = -1e9
    failures = frames = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            failures += 1
            if failures > 60:
                raise CameraError("อ่านภาพจากกล้องไม่ได้ต่อเนื่อง")
            continue
        failures = 0
        frame = np.ascontiguousarray(frame[:, ::-1])                       # กลับซ้ายขวาเหมือนกระจก
        t = clock()
        pose = detect(frame[:, :, ::-1], int(t * 1000))
        result = analyzer.update(pose, t)
        if result.event is not None:
            last_event, last_event_t = result.event, t
            if beeper and result.event.issues:
                beeper()
        draw_pose(frame, pose, analyzer.cfg.min_visibility)
        draw_border(frame, *border_style(result, last_event, t - last_event_t))
        draw_hud(frame, result, analyzer, last_event, t - last_event_t, font_path)
        key = show(frame)
        if key in (ord("q"), 27):
            break
        if key in (ord("r"), ord("R")):
            analyzer.reset()
            last_event = None
        if key in (ord("c"), ord("C")):
            analyzer.recalibrate()
        frames += 1
        if max_frames is not None and frames >= max_frames:
            break
    return analyzer.summary()


def save_log(summary: dict, log_dir: Path = LOG_DIR) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"squat_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), **summary},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--set ต้องอยู่ในรูป ชื่อ=ค่า (ได้รับ '{p}')")
        k, v = p.split("=", 1)
        out[k.strip()] = int(v) if v.strip().lstrip("-").isdigit() else float(v)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="นับ Squat และเตือนท่าผิดแบบเรียลไทม์")
    ap.add_argument("--view", choices=["front", "side"], default="front", help="มุมกล้อง (ค่าเริ่มต้น front)")
    ap.add_argument("--camera", type=int, default=0, help="หมายเลขกล้อง")
    ap.add_argument("--no-sound", action="store_true", help="ปิดเสียงเตือน")
    ap.add_argument("--set", action="append", default=[], metavar="ชื่อ=ค่า", help="ปรับเกณฑ์ เช่น --set side_count=130")
    args = ap.parse_args(argv)

    import cv2

    try:
        cfg = SquatConfig.from_overrides(view=args.view, **parse_overrides(args.set))
    except ValueError as e:
        raise SystemExit(str(e))
    analyzer = SquatAnalyzer(cfg)
    detector = LiveDetector()
    cap = open_camera(args.camera)

    def show(frame: np.ndarray) -> int:
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:       # กดปิดหน้าต่างด้วย X
            return ord("q")
        return -1 if key == 255 else key

    print(f"เริ่มแล้ว (มุมกล้อง: {args.view}) ยืนห่างกล้องประมาณ 2-3 เมตร ให้เห็นตั้งแต่ศีรษะถึงเท้า | Q = จบ")
    try:
        summary = run_live(cap, detector, analyzer, show, None if args.no_sound else make_beeper(),
                           font_path=find_thai_font())
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
    path = save_log(summary)
    print(f"\nสรุป: นับได้ {summary['reps']} ครั้ง (พยายาม {summary['attempts']} ครั้ง) ข้อเตือน: {summary['issue_counts'] or 'ไม่มี'}")
    print(f"บันทึกสรุปไว้ที่ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())