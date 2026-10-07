"""
scanner.py - Phase 1: AI Somatotype Scanner ของ FitMetrix AI

ขั้นตอน:
  ภาพถ่ายยืนตรง -> MediaPipe PoseLandmarker (ตำแหน่งข้อต่อ + เงาร่างกาย/segmentation mask)
  -> วัดความกว้าง ไหล่ / เอว (แคบสุด) / สะโพก (กว้างสุด) จาก "เงาร่างกาย" หลังตัดแขนออก
  -> จำแนก 2 ทาง: (1) จากตัวเลข BMI + %ไขมัน  (2) จากสัดส่วนในภาพ  แล้วรวมหลักฐานทั้งสอง

ทำไมไม่ใช้ตำแหน่งข้อต่อสะโพกวัดความกว้างโดยตรง:
  จุดข้อต่อสะโพกของ MediaPipe คือ "ศูนย์กลางข้อต่อ" แคบกว่าสะโพกจริงมาก (โดยเฉพาะคนที่มีไขมัน/กล้ามเนื้อสะโพกมาก)
  จึงใช้ข้อต่อเพียงเพื่อหา "ระดับความสูง" แล้ววัดความกว้างจริงจากเงาร่างกายแทน

ข้อจำกัดที่ต้องรู้ (แสดงให้ผู้ใช้เห็นในแอปด้วย):
  - เกณฑ์ทุกตัวเป็นค่าประมาณจากสัดส่วนร่างกายทั่วไป ยังไม่ได้สอบเทียบกับภาพที่ติดป้ายจริง ปรับได้ที่ค่าคงที่ด้านล่าง
  - เงาจากภาพ 2 มิติถูกกระทบจากชุดที่หลวม ท่ายืน มุมกล้อง และคุณภาพ mask
  - ระดับไขมันไม่ได้เดาจากภาพ ใช้สูตร Navy (มีรอบเอว/คอ) หรือ Deurenberg (จาก BMI)
  - ผลเป็นการประมาณ ไม่ใช่การวินิจฉัยทางการแพทย์
"""

from __future__ import annotations

import math
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import calc_bmi  # noqa: E402
from src.metabolic import ProfileError  # noqa: E402

MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# ดัชนีจุดข้อต่อของ MediaPipe Pose (33 จุด)
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HAND, R_HAND = (17, 19, 21), (18, 20, 22)      # นิ้วก้อย / นิ้วชี้ / นิ้วโป้ง
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# ---------- เกณฑ์คุณภาพภาพ ----------
MIN_VISIBILITY = 0.5
MIN_SHOULDER_WIDTH_FRAC = 0.08      # ไหล่ต้องกว้างอย่างน้อย 8% ของความกว้างภาพ
MAX_TILT_DEG = 8.0                  # ไหล่/สะโพกเอียงเกินนี้ให้เตือน
MAX_DEPTH_RATIO = 0.45              # ไหล่สองข้างลึกต่างกันมาก = ตัวหันเฉียง
MASK_THRESHOLD = 0.5
MIN_BODY_HEIGHT_FRAC = 0.5          # เงาร่างกายสูงอย่างน้อย 50% ของภาพ
EDGE_MARGIN_FRAC = 0.01             # ศีรษะ/เท้าชิดขอบภาพ = ถูกตัด วัดสัดส่วนไม่ได้
ARM_RADIUS_FRAC = 0.035             # รัศมีแขนที่ตัดออกจากเงา (สัดส่วนของส่วนสูงตัว)
SHOULDER_MAX_JOINT_RATIO = 1.6      # ความกว้างไหล่จากเงา / ระยะข้อต่อไหล่ เกินนี้ = มีแขนที่ยกสูงติดเงา ไม่ใช้ค่าไหล่

# ---------- เกณฑ์จากตัวเลข (ค่าประมาณ ปรับได้) ----------
BF_LEAN = {"male": 14.0, "female": 21.0}   # %ไขมัน ต่ำกว่านี้ = ผอม/เล็ก
BF_HIGH = {"male": 24.0, "female": 31.0}   # %ไขมัน สูงกว่านี้ = เก็บไขมันมาก
BMI_LOW = 18.5
BMI_LEAN_MAX = 22.5

# ---------- เกณฑ์จากเงาในภาพ (ค่าประมาณจากสัดส่วนร่างกายทั่วไป ยังไม่ได้สอบเทียบ) ----------
# ความกว้างเงา / ส่วนสูงตัว (หลังตัดแขน)
WAIST_LEAN = {"male": 0.150, "female": 0.150}
WAIST_FULL = {"male": 0.200, "female": 0.200}
HIP_LEAN = {"male": 0.180, "female": 0.205}
HIP_FULL = {"male": 0.230, "female": 0.260}
SHAPE_THRESHOLDS = {"male": (1.10, 1.35), "female": (0.95, 1.20)}   # ไหล่:สะโพก (pear / balanced / v_taper)

SOMATOTYPE_LABELS_TH = {
    "ectomorph": "Ectomorph (ผอมเพรียว เผาผลาญไว)",
    "mesomorph": "Mesomorph (สมส่วน กล้ามเนื้อพัฒนาง่าย)",
    "endomorph": "Endomorph (โครงใหญ่ เก็บไขมันง่าย)",
}
SOMATOTYPE_SHORT_TH = {"ectomorph": "Ectomorph", "mesomorph": "Mesomorph", "endomorph": "Endomorph"}
SHAPE_LABELS_TH = {
    "v_taper": "ไหล่กว้างกว่าสะโพกชัดเจน (V-Taper)",
    "balanced": "ไหล่และสะโพกสมดุล",
    "pear": "สะโพกกว้างกว่าไหล่ (Pear)",
}
NO_PERSON_MESSAGE = "ไม่พบคนในภาพ ลองยืนให้เห็นเต็มตัวในที่ที่มีแสงพอ"


class ScannerSetupError(RuntimeError):
    """MediaPipe ไม่พร้อมใช้งาน (ไม่ได้ติดตั้ง หรือยังไม่มีไฟล์โมเดล)"""


class Landmark(NamedTuple):
    x: float            # พิกัด normalized 0-1 (กว้าง)
    y: float            # พิกัด normalized 0-1 (สูง)
    z: float
    visibility: float


class PoseDetection(NamedTuple):
    landmarks: list                 # list[Landmark] 33 จุด
    image_size: tuple               # (กว้าง, สูง) เป็นพิกเซล
    mask: Optional[Any] = None      # เงาร่างกาย ค่า 0-1 ขนาด สูง x กว้าง (ไม่มีก็ได้)


# ----------------------------- โมเดลและ detector ----------------------------
def download_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """ดาวน์โหลดไฟล์โมเดลท่าทาง (lite ประมาณ 5-6 MB) ถ้ายังไม่มี"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size < 1_000_000:
            raise OSError("ไฟล์ที่ดาวน์โหลดมีขนาดเล็กผิดปกติ")
        tmp.replace(path)
    except Exception as e:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        raise ScannerSetupError(f"ดาวน์โหลดโมเดลไม่สำเร็จ ({e}) ลองดาวน์โหลดเองจาก {url} แล้ววางที่ {path}") from e
    return path

def _crop_to_multiple_of_4(image_rgb: np.ndarray) -> np.ndarray:
    """ตัดคอลัมน์ขวาสุดทิ้งไม่เกิน 3 พิกเซล ให้ความกว้างหาร 4 ลงตัว (ดูเหตุผลที่ _mask_from_result)"""
    h, w = image_rgb.shape[:2]
    return image_rgb if w % 4 == 0 else np.ascontiguousarray(image_rgb[:, : w - (w % 4)])


def _mask_from_result(result) -> Optional[np.ndarray]:
    """
    ดึงเงาร่างกายจากผลของ PoseLandmarker เป็น numpy (คัดลอกข้อมูลออกมา) หรือ None ถ้าดึงไม่ได้
    ข้อควรระวังสองข้อ (ทดสอบยืนยันแล้ว):
      1) numpy_view() ของภาพ float32 ใน MediaPipe ทำให้โปรแกรมทั้งตัวดับทันที (native abort
         "Check failed: 1 == ChannelSize()") เมื่อความกว้างภาพไม่ใช่ผลคูณของ 4 try/except จับไม่ได้
         จึงต้องเช็คความกว้างก่อนเรียก
      2) numpy_view() เป็นแค่ "มุมมอง" ของหน่วยความจำที่เป็นของภาพ พอ result ถูกทำลายข้อมูลจะเพี้ยน
         จึงต้องคัดลอกก่อนส่งออกจากฟังก์ชันนี้
    """
    masks = getattr(result, "segmentation_masks", None)
    if not masks:
        return None
    image = masks[0]
    if image.width % 4 != 0:
        return None
    try:
        arr = np.squeeze(np.array(image.numpy_view(), dtype=np.float32))   # np.array = คัดลอก
    except Exception:  # noqa: BLE001
        return None
    return arr if arr.ndim == 2 else None

class PoseDetector:
    """ครอบ MediaPipe PoseLandmarker (Tasks API) รับภาพ RGB คืนจุดข้อต่อ 33 จุด + เงาร่างกาย"""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not Path(model_path).exists():
            raise ScannerSetupError(f"ไม่พบไฟล์โมเดล {model_path} (รัน python setup_scanner.py เพื่อดาวน์โหลด)")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise ScannerSetupError("ยังไม่ได้ติดตั้ง mediapipe (pip install mediapipe)") from e

        self._mp = mp
        # อ่านเป็น buffer เพื่อเลี่ยงปัญหา path ที่มีอักษรที่ไม่ใช่ ASCII บน Windows
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            output_segmentation_masks=True,       # ต้องใช้เงาร่างกายวัดความกว้างเอว/สะโพก
        )
        try:
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:  # noqa: BLE001  (ไฟล์โมเดลเสียหาย/ไม่สมบูรณ์)
            raise ScannerSetupError(f"โหลดไฟล์โมเดลไม่ได้ ({e}) ลองลบ {model_path} แล้วรัน python setup_scanner.py ใหม่") from e

    def detect(self, image_rgb) -> Optional[PoseDetection]:
        image_rgb = _crop_to_multiple_of_4(np.asarray(image_rgb))      # image_size ที่คืนไปคือขนาดหลังตัด
        image_rgb = np.ascontiguousarray(image_rgb)
        h, w = image_rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        landmarks = [Landmark(p.x, p.y, p.z, p.visibility if p.visibility is not None else 0.0)
                     for p in result.pose_landmarks[0]]
        return PoseDetection(landmarks=landmarks, image_size=(w, h), mask=_mask_from_result(result))

    def close(self) -> None:
        self._landmarker.close()


# ----------------------------- คุณภาพท่าทาง ---------------------------------
def _tilt_deg(a: Landmark, b: Landmark, w: int, h: int) -> float:
    ang = abs(math.degrees(math.atan2((a.y - b.y) * h, (a.x - b.x) * w)))
    return min(ang, 180 - ang)   # 0 = เส้นอยู่ในแนวนอนพอดี


def compute_pose_metrics(landmarks: list, image_w: int, image_h: int) -> dict:
    ls, rs, lh, rh = (landmarks[i] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))
    shoulder_w_norm = math.hypot((ls.x - rs.x) * image_w, (ls.y - rs.y) * image_h) / image_w
    return {
        "shoulder_width_frac": round(shoulder_w_norm, 3),
        "shoulder_tilt_deg": round(_tilt_deg(ls, rs, image_w, image_h), 1),
        "hip_tilt_deg": round(_tilt_deg(lh, rh, image_w, image_h), 1),
        "depth_ratio": round(abs(ls.z - rs.z) / max(shoulder_w_norm, 1e-6), 2),
    }


def assess_pose_quality(landmarks: list, metrics: dict) -> tuple[list[str], list[str]]:
    """คืน (errors ที่ใช้ภาพไม่ได้, warnings ที่ใช้ได้แต่ผลอาจคลาดเคลื่อน)"""
    errors, warnings = [], []

    core = [landmarks[i] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
    if min(p.visibility for p in core) < MIN_VISIBILITY:
        errors.append("มองเห็นไหล่หรือสะโพกไม่ชัด ลองใส่ชุดกระชับ เพิ่มแสง และให้เห็นลำตัวเต็ม")
    if metrics["shoulder_width_frac"] < MIN_SHOULDER_WIDTH_FRAC:
        errors.append("ตัวเล็กเกินไปในภาพ ลองเข้าใกล้กล้องอีกนิด")

    lower = [landmarks[i] for i in (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)]
    if min(p.visibility for p in lower) < MIN_VISIBILITY:
        warnings.append("เห็นร่างกายไม่เต็มตัว (เข่า/ข้อเท้าไม่ชัด) ควรให้เห็นตั้งแต่ศีรษะถึงเท้า")
    if any(not (0 <= p.x <= 1 and 0 <= p.y <= 1) for p in core + lower):
        warnings.append("ร่างกายบางส่วนอยู่นอกกรอบภาพ")
    if metrics["shoulder_tilt_deg"] > MAX_TILT_DEG or metrics["hip_tilt_deg"] > MAX_TILT_DEG:
        warnings.append("ไหล่หรือสะโพกเอียง ลองยืนตรงและให้กล้องอยู่ระดับเอว")
    if metrics["depth_ratio"] > MAX_DEPTH_RATIO:
        warnings.append("ตัวอาจไม่หันตรงเข้ากล้อง สัดส่วนที่ได้อาจคลาดเคลื่อน")
    return errors, warnings


# ----------------------------- วัดความกว้างจากเงาร่างกาย ----------------------
def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """ช่วงพิกเซลที่ติดกันในแถวเดียว -> [(เริ่ม, จบ), ...]"""
    padded = np.concatenate(([0], row.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2].tolist(), (edges[1::2] - 1).tolist()))


def _central_extent(row: np.ndarray, x_mid: float) -> Optional[tuple[int, int]]:
    """ช่วงของลำตัวที่ผ่านเส้นกึ่งกลาง ถ้าเส้นกลางตกในช่องว่าง (เช่นระหว่างขา) ใช้ขอบนอกของสองข้างที่ใกล้ที่สุด"""
    runs = _runs(row)
    if not runs:
        return None
    xm = int(round(x_mid))
    for s, e in runs:
        if s <= xm <= e:
            return s, e
    left = [r for r in runs if r[1] < xm]
    right = [r for r in runs if r[0] > xm]
    if left and right:
        return left[-1][0], right[0][1]
    return None


def _arm_paths(landmarks: list, w: int, h: int) -> tuple[list, list[str]]:
    """เส้นกลางแขนแต่ละข้าง: ไหล่ -> ศอก -> ข้อมือ -> ฝ่ามือ (พิกเซล) และชื่อแขนที่มองไม่เห็น"""
    paths, missing = [], []
    for name, ids, hand in (("ซ้าย", (L_SHOULDER, L_ELBOW, L_WRIST), L_HAND),
                            ("ขวา", (R_SHOULDER, R_ELBOW, R_WRIST), R_HAND)):
        pts = [landmarks[i] for i in ids]
        if min(p.visibility for p in pts) < MIN_VISIBILITY:
            missing.append(name)
            continue
        path = [(p.x * w, p.y * h) for p in pts]
        hands = [landmarks[i] for i in hand if landmarks[i].visibility >= MIN_VISIBILITY]
        if hands:
            path.append((float(np.mean([p.x for p in hands])) * w, float(np.mean([p.y for p in hands])) * h))
        paths.append(path)
    return paths, missing


def _arm_mask(shape: tuple[int, int], paths: list, radius: float) -> np.ndarray:
    from PIL import Image, ImageDraw

    h, w = shape
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for path in paths:
        draw.line(path, fill=255, width=max(1, int(round(2 * radius))), joint="curve")
        for x, y in path:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)
    return np.asarray(img) > 0


def measure_silhouette(landmarks: list, mask, threshold: float = MASK_THRESHOLD) -> dict:
    """
    วัดความกว้างจากเงาร่างกาย (หน่วยพิกเซลของ mask) ระดับความสูงอ้างอิงจากข้อต่อไหล่/สะโพก
      ไหล่  = กว้างสุดบริเวณระดับไหล่ (รวมกล้ามไหล่)
      เอว   = แคบสุดระหว่าง 55-85% ของระยะไหล่->สะโพก (หลังตัดแขน)
      สะโพก = กว้างสุดบริเวณข้อต่อสะโพกลงมา (หลังตัดแขน)
    แขนถูกตัดออกด้วยแคปซูลรัศมี ARM_RADIUS_FRAC x ส่วนสูงตัว ตามแนวไหล่-ศอก-ข้อมือ-ฝ่ามือ
    เพราะเวลาแขนแนบลำตัว เงาของแขนจะรวมกับลำตัวและทำให้เอว/สะโพกดูกว้างเกินจริง
    คืน dict ที่มี ok / problems และค่าที่วัดได้ (ถ้า ok = False อย่านำไปจำแนก)
    """
    binary = np.asarray(mask) >= threshold
    h, w = binary.shape
    problems: list[str] = []
    out: dict = {"ok": False, "problems": problems}

    rows = np.flatnonzero(binary.any(axis=1))
    if rows.size == 0:
        problems.append("ไม่พบเงาร่างกายในภาพ")
        return out
    ls, rs, lh, rh = (landmarks[i] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))

    # ยอดศีรษะ: ดูเฉพาะแถบคอลัมน์รอบจมูก เพื่อไม่ให้มือ/แขนที่ยกสูงมาเป็น "ยอด" ของตัว
    top, bottom = int(rows[0]), int(rows[-1])
    nose = landmarks[NOSE]
    if nose.visibility >= MIN_VISIBILITY:
        half = 0.6 * abs(ls.x - rs.x) * w
        c0, c1 = max(0, int(nose.x * w - half)), min(w, int(nose.x * w + half) + 1)
        head_rows = np.flatnonzero(binary[:, c0:c1].any(axis=1))
        if head_rows.size:
            top = int(head_rows[0])
    body_h = bottom - top + 1
    if body_h < MIN_BODY_HEIGHT_FRAC * h:
        problems.append("ตัวเล็กเกินไปในภาพ ลองถอยกล้องให้เห็นเต็มตัวและเข้าใกล้ให้พอดีกรอบ")
        return out
    head_cut = top <= EDGE_MARGIN_FRAC * h
    feet_cut = bottom >= (1 - EDGE_MARGIN_FRAC) * h - 1
    if head_cut:
        problems.append("ศีรษะ (รวมผม) ชิดหรือถูกตัดขอบภาพด้านบน ต้องเหลือที่ว่างเหนือศีรษะ ลองถอยกล้องหรือเงยกล้องขึ้นเล็กน้อย")
    if feet_cut:
        problems.append("เท้าชิดหรือถูกตัดขอบภาพด้านล่าง ต้องให้เห็นเท้าครบ ลองถอยกล้องออกอีกนิด")
    if head_cut or feet_cut:
        return out

    y_sh, y_hp = (ls.y + rs.y) / 2 * h, (lh.y + rh.y) / 2 * h
    torso_len = y_hp - y_sh
    if torso_len < 0.05 * body_h:
        problems.append("ตำแหน่งไหล่/สะโพกที่ตรวจจับได้ผิดปกติ")
        return out
    x_mid = float(np.mean([ls.x, rs.x, lh.x, rh.x])) * w

    paths, missing_arms = _arm_paths(landmarks, w, h)
    radius = ARM_RADIUS_FRAC * body_h
    torso = binary & ~_arm_mask((h, w), paths, radius) if paths else binary
    if missing_arms:
        problems.append(f"มองไม่เห็นแขน{' และ '.join(missing_arms)} สัดส่วนที่วัดอาจคลาดเคลื่อน")

        # มือ/แขนอยู่หน้าลำตัวช่วงเอว-สะโพก (เช่นวางมือที่เอว): เงาแขนซ้อนลำตัว การตัดแขนจะกินลำตัวไปด้วย วัดไม่ได้
    hx0, hx1 = sorted((lh.x * w, rh.x * w))
    pad = 0.5 * (hx1 - hx0)
    zone_top, zone_bot = y_sh + 0.50 * torso_len, y_hp + 0.09 * body_h
    for path in paths:
        for px, py in (path[2], path[-1]):                       # ข้อมือ และปลายมือ
            if hx0 - pad <= px <= hx1 + pad and zone_top <= py <= zone_bot:
                problems.append("มือหรือแขนอยู่หน้าลำตัวบริเวณเอว/สะโพก (เช่นวางมือที่เอว) บังเงาลำตัว วัดเอวและสะโพกไม่ได้ "
                                "ให้ปล่อยแขนลงข้างลำตัวห่างเล็กน้อย")
                return out

    def rows_between(a: float, b: float) -> range:
        return range(max(0, int(a)), min(h - 1, int(b)) + 1)

    def best(mask_: np.ndarray, band: range, pick) -> Optional[tuple[int, int, int, int]]:
        """(ความกว้าง, แถว, ซ้าย, ขวา) ของแถวที่ pick (max/min) เลือกจากความกว้างลำตัวส่วนกลาง"""
        cands = []
        for y in band:
            ext = _central_extent(mask_[y], x_mid)
            if ext is not None:
                cands.append((ext[1] - ext[0] + 1, y, ext[0], ext[1]))
        return pick(cands, key=lambda c: c[0]) if cands else None

    shoulder = best(binary, rows_between(y_sh - 0.02 * body_h, y_sh + 0.03 * body_h), max)
    waist = best(torso, rows_between(y_sh + 0.55 * torso_len, y_sh + 0.85 * torso_len), min)
    hip = best(torso, rows_between(y_hp - 0.03 * body_h, y_hp + 0.07 * body_h), max)
    if shoulder is None or waist is None or hip is None:
        problems.append("วัดความกว้างลำตัวจากเงาไม่ได้ (เงาขาดหาย) ลองถ่ายใหม่ในที่ที่ฉากหลังเรียบและแสงพอ")
        return out

    shoulder_w, waist_w, hip_w = shoulder[0], waist[0], hip[0]
    waist_ratio, hip_ratio = waist_w / body_h, hip_w / body_h
        # ไหล่จริงกว้างไม่เกิน ~1.5 เท่าของระยะข้อต่อไหล่ ถ้าเกินมากแปลว่าแขนที่ยกสูง/งอข้างไหล่ติดมากับเงา
    joint_w = math.hypot((ls.x - rs.x) * w, (ls.y - rs.y) * h)
    shoulder_ok = shoulder_w / max(joint_w, 1.0) <= SHOULDER_MAX_JOINT_RATIO
    if not shoulder_ok:
        problems.append("แขนที่ยกสูงหรืองอข้างไหล่ติดมากับเงา จึงวัดความกว้างไหล่ไม่ได้ (ไม่กระทบการจำแนกจากเอว/สะโพก)")
    if not (0.07 <= waist_ratio <= 0.45 and 0.10 <= hip_ratio <= 0.50):
        problems.append("ค่าที่วัดจากเงาผิดปกติ (อาจเกิดจากฉากหลังรก หรือมีคนอื่น/วัตถุติดเงา) ลองถ่ายใหม่")
        return out

    norm = lambda y, x0, x1: (round(y / h, 4), round(x0 / w, 4), round((x1 + 1) / w, 4))  # noqa: E731
    out.update({
        "ok": True,
        "body_height_px": body_h,
        "shoulder_width_px": shoulder_w if shoulder_ok else None, "waist_width_px": waist_w, "hip_width_px": hip_w,
        "shoulder_ratio": round(shoulder_w / body_h, 3) if shoulder_ok else None,
        "waist_ratio": round(waist_ratio, 3),
        "hip_ratio": round(hip_ratio, 3),
        "waist_hip_ratio": round(waist_w / hip_w, 3),
        "shoulder_hip_ratio": round(shoulder_w / hip_w, 3) if shoulder_ok else None,
        "arms_removed": len(paths),
        "arms_missing": missing_arms,
        # พิกัด normalized ไว้วาดเส้นวัดบนภาพ (y, x ซ้าย, x ขวา)
        "lines": {"shoulder": norm(*shoulder[1:]) if shoulder_ok else None, "waist": norm(*waist[1:]), "hip": norm(*hip[1:])},
        "arm_paths": [[(round(x / w, 4), round(y / h, 4)) for x, y in p] for p in paths],
        "arm_radius_frac": round(radius / h, 4),
    })
    return out


def describe_shape(gender: str, shoulder_hip_ratio: float) -> str:
    lo, hi = SHAPE_THRESHOLDS[gender.lower()]
    return "v_taper" if shoulder_hip_ratio >= hi else "balanced" if shoulder_hip_ratio >= lo else "pear"


# ----------------------------- ระดับไขมันในร่างกาย ----------------------------
def estimate_body_fat(
    gender: str, age: float, height_cm: float, weight_kg: float,
    waist_cm: Optional[float] = None, neck_cm: Optional[float] = None, hip_cm: Optional[float] = None,
) -> tuple[float, str]:
    """
    คืน (%ไขมัน, วิธีที่ใช้)
      - "navy"        : สูตร US Navy (ต้องมีรอบเอวและรอบคอ หญิงต้องมีรอบสะโพกด้วย)
      - "deurenberg"  : สูตรจาก BMI + อายุ + เพศ (ใช้เมื่อไม่มีรอบตัว ความแม่นยำต่ำกว่า)
    """
    g = gender.lower()
    if waist_cm and neck_cm:
        try:
            if g == "male" and waist_cm > neck_cm:
                bf = 495 / (1.0324 - 0.19077 * math.log10(waist_cm - neck_cm) + 0.15456 * math.log10(height_cm)) - 450
                return round(min(max(bf, 3.0), 60.0), 1), "navy"
            if g == "female" and hip_cm and waist_cm + hip_cm > neck_cm:
                bf = 495 / (1.29579 - 0.35004 * math.log10(waist_cm + hip_cm - neck_cm) + 0.22100 * math.log10(height_cm)) - 450
                return round(min(max(bf, 3.0), 60.0), 1), "navy"
        except (ValueError, ZeroDivisionError):
            pass  # ตกลงไปใช้สูตร Deurenberg
    bmi = calc_bmi(weight_kg, height_cm)
    bf = 1.20 * bmi + 0.23 * age - 10.8 * (1 if g == "male" else 0) - 5.4
    return round(min(max(bf, 3.0), 60.0), 1), "deurenberg"


# ----------------------------- จำแนก Somatotype ------------------------------
ORDER = {"ectomorph": 0, "mesomorph": 1, "endomorph": 2}


def classify_somatotype(gender: str, bmi: float, body_fat_pct: float) -> str:
    """
    จากตัวเลข (ค่าประมาณ):
      Endomorph : %ไขมัน >= BF_HIGH
      Ectomorph : BMI < 18.5  หรือ  (%ไขมัน < BF_LEAN และ BMI < 22.5)
      Mesomorph : นอกเหนือจากนั้น
    """
    g = gender.lower()
    if body_fat_pct >= BF_HIGH[g]:
        return "endomorph"
    if bmi < BMI_LOW or (body_fat_pct < BF_LEAN[g] and bmi < BMI_LEAN_MAX):
        return "ectomorph"
    return "mesomorph"


def classify_visual(gender: str, waist_ratio: float, hip_ratio: float) -> str:
    """
    จากสัดส่วนเงา (ความกว้าง/ส่วนสูงตัว):
      Endomorph : เอวกว้าง >= WAIST_FULL  หรือ  สะโพกกว้าง >= HIP_FULL
      Ectomorph : เอวแคบ < WAIST_LEAN  และ  สะโพกแคบ < HIP_LEAN
      Mesomorph : นอกเหนือจากนั้น
    """
    g = gender.lower()
    if waist_ratio >= WAIST_FULL[g] or hip_ratio >= HIP_FULL[g]:
        return "endomorph"
    if waist_ratio < WAIST_LEAN[g] and hip_ratio < HIP_LEAN[g]:
        return "ectomorph"
    return "mesomorph"


def combine_evidence(numeric: str, visual: Optional[str], measured_body_fat: bool) -> tuple[str, str, list[str]]:
    """
    รวมหลักฐาน 2 ทาง คืน (ผลสรุป, agreement, เหตุผล)
      agreement: "numbers_only" | "agree" | "conflict"
    กรณีขัดแย้งห่างกัน 1 ระดับ: ถ้ามีรอบเอว/คอที่วัดจริง (Navy) เชื่อตัวเลข ไม่งั้นเชื่อรูปร่างในภาพ
    (เพราะ %ไขมันจากสูตร BMI เป็นค่าประมาณ ส่วนเงาคือสิ่งที่วัดได้จริงจากภาพ)
    กรณีห่างกัน 2 ระดับ: จัดเป็น Mesomorph และเตือนให้ตรวจสอบ
    """
    if visual is None:
        return numeric, "numbers_only", []
    if visual == numeric:
        return numeric, "agree", ["ผลจากตัวเลขและสัดส่วนในภาพตรงกัน"]
    if abs(ORDER[numeric] - ORDER[visual]) == 2:
        return "mesomorph", "conflict", [
            f"ตัวเลขชี้ไปทาง {SOMATOTYPE_SHORT_TH[numeric]} แต่ภาพชี้ไปทาง {SOMATOTYPE_SHORT_TH[visual]} "
            "ขัดแย้งกันมาก จึงจัดเป็น Mesomorph (กลาง) ควรตรวจสอบข้อมูลที่กรอกและวิธีถ่ายภาพ"]
    if measured_body_fat:
        return numeric, "conflict", [
            f"ภาพชี้ไปทาง {SOMATOTYPE_SHORT_TH[visual]} แต่ตัวเลขที่ใช้รอบเอว/คอที่วัดจริงชี้ไปทาง "
            f"{SOMATOTYPE_SHORT_TH[numeric]} จึงเชื่อตัวเลข"]
    return visual, "conflict", [
        f"ตัวเลข (BMI) ชี้ไปทาง {SOMATOTYPE_SHORT_TH[numeric]} แต่รูปร่างในภาพชี้ไปทาง {SOMATOTYPE_SHORT_TH[visual]} "
        "จึงให้น้ำหนักกับรูปร่างในภาพมากกว่า เพราะ %ไขมันจากสูตร BMI เป็นแค่ค่าประมาณ"]


# ----------------------------- ผลลัพธ์และฟังก์ชันหลัก -------------------------
@dataclass
class ScanResult:
    somatotype: str
    somatotype_label: str
    confidence: str                          # "medium" | "low" (ไม่มี "high" เพราะเป็นการประมาณ)
    agreement: str                           # "numbers_only" | "agree" | "conflict"
    numeric_somatotype: str
    visual_somatotype: Optional[str]
    bmi: float
    body_fat_pct: float
    body_fat_method: str
    photo_used: bool
    shape: Optional[str] = None
    shape_label: Optional[str] = None
    silhouette: dict = field(default_factory=dict)
    pose_metrics: dict = field(default_factory=dict)
    photo_errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _validate(gender, age, height_cm, weight_kg) -> str:
    errors = []
    g = str(gender).strip().lower()
    if g not in ("male", "female"):
        errors.append("เพศต้องเป็น male หรือ female")
    for label, v, lo, hi in [("อายุ", age, 18, 100), ("ส่วนสูง", height_cm, 100, 250), ("น้ำหนัก", weight_kg, 30, 250)]:
        try:
            if not lo <= float(v) <= hi:
                errors.append(f"{label} ต้องอยู่ระหว่าง {lo}-{hi}" + (" (ระบบนี้ให้คำแนะนำเฉพาะผู้ใหญ่)" if label == "อายุ" else ""))
        except (TypeError, ValueError):
            errors.append(f"{label} ต้องเป็นตัวเลข")
    if errors:
        raise ProfileError(errors)
    return g


def analyze_body(
    gender: str, age: float, height_cm: float, weight_kg: float,
    detection: Optional[PoseDetection] = None,
    waist_cm: Optional[float] = None, neck_cm: Optional[float] = None, hip_cm: Optional[float] = None,
    photo_errors: Optional[list[str]] = None,
) -> ScanResult:
    """จำแนก Somatotype จากข้อมูลตัวเลข + (ถ้ามี) เงาร่างกายจากภาพ แล้วรวมหลักฐานทั้งสองทาง"""
    g = _validate(gender, age, height_cm, weight_kg)
    warnings: list[str] = []
    photo_errors = list(photo_errors or [])
    pose_metrics: dict = {}
    sil: dict = {}
    visual: Optional[str] = None
    shape = None

    # ---- จากภาพ ----
    if detection is not None:
        w, h = detection.image_size
        pose_metrics = compute_pose_metrics(detection.landmarks, w, h)
        errs, warns = assess_pose_quality(detection.landmarks, pose_metrics)
        if errs:
            photo_errors += errs
        else:
            warnings += warns
            if detection.mask is None:
                warnings.append("ไม่ได้รับข้อมูลเงาร่างกายจากโมเดล จึงวัดสัดส่วนจากภาพไม่ได้")
            else:
                sil = measure_silhouette(detection.landmarks, detection.mask)
                if sil["ok"]:
                    visual = classify_visual(g, sil["waist_ratio"], sil["hip_ratio"])
                    shape = describe_shape(g, sil["shoulder_hip_ratio"]) if sil["shoulder_hip_ratio"] is not None else None
                    warnings += sil["problems"]
                    if sil["arms_missing"] and visual == "endomorph":
                        # แขนที่หักออกไม่ได้อาจติดเงาลำตัว ทำให้เอว/สะโพกกว้างเกินจริง ภาพจึงชี้ว่าอวบเกินจริงได้
                        # (วัดผิดได้แต่ด้าน "กว้างเกิน" ไม่มีด้าน "แคบเกิน") จึงไม่ใช้ภาพตัดสินว่า Endomorph
                        warnings.append("มองไม่เห็นแขนครบทั้งสองข้าง เงาแขนอาจติดลำตัวทำให้วัดกว้างเกินจริง จึงไม่ใช้ภาพนี้ตัดสินว่า Endomorph "
                                        "ลองถ่ายใหม่ให้เห็นแขนทั้งสองข้างชัดเจน")
                        visual, shape = None, None
                else:
                    photo_errors += sil["problems"]

    # ---- จากตัวเลข ----
    bmi = calc_bmi(weight_kg, height_cm)
    bf, method = estimate_body_fat(g, age, height_cm, weight_kg, waist_cm, neck_cm, hip_cm)
    numeric = classify_somatotype(g, bmi, bf)

    # ---- รวมหลักฐาน ----
    final, agreement, combine_reasons = combine_evidence(numeric, visual, measured_body_fat=(method == "navy"))

    reasons = [
        f"BMI {bmi:.1f}",
        f"ไขมันในร่างกายโดยประมาณ {bf}% ("
        + ("สูตร US Navy จากรอบเอว/คอที่วัดจริง" if method == "navy" else "สูตร Deurenberg จาก BMI อายุ เพศ ความแม่นยำจำกัด") + ")",
        f"จากตัวเลข: {SOMATOTYPE_SHORT_TH[numeric]}",
    ]
    if visual:
        reasons.append(
            f"จากรูปร่างในภาพ: {SOMATOTYPE_SHORT_TH[visual]} (เอว {sil['waist_ratio']:.3f} / สะโพก {sil['hip_ratio']:.3f} "
            f"ของส่วนสูง, เอว:สะโพก {sil['waist_hip_ratio']}" + (f", {SHAPE_LABELS_TH[shape]}" if shape else "") + ")")
    reasons += combine_reasons

    if visual is None:
        warnings.append("ผลนี้ใช้ข้อมูลตัวเลขเท่านั้น ไม่ได้ใช้สัดส่วนจากภาพ")
    confidence = "medium" if agreement == "agree" else "low"   # ไม่ให้ "สูง" เพราะทุกขั้นเป็นการประมาณ

    return ScanResult(
        somatotype=final, somatotype_label=SOMATOTYPE_LABELS_TH[final], confidence=confidence,
        agreement=agreement, numeric_somatotype=numeric, visual_somatotype=visual,
        bmi=round(bmi, 1), body_fat_pct=bf, body_fat_method=method, photo_used=visual is not None,
        shape=shape, shape_label=SHAPE_LABELS_TH.get(shape) if shape else None,
        silhouette=sil if sil.get("ok") else {}, pose_metrics=pose_metrics,
        photo_errors=photo_errors, warnings=warnings, reasons=reasons,
    )


def analyze_detection(detection: Optional[PoseDetection], gender, age, height_cm, weight_kg, **measurements) -> ScanResult:
    """วิเคราะห์ผลการตรวจจับ ถ้า detection เป็น None (ไม่พบคน) จะใช้ข้อมูลตัวเลขอย่างเดียวพร้อมแจ้งเตือน"""
    if detection is None:
        return analyze_body(gender, age, height_cm, weight_kg, photo_errors=[NO_PERSON_MESSAGE], **measurements)
    return analyze_body(gender, age, height_cm, weight_kg, detection=detection, **measurements)


def scan_image(image_rgb, detector, gender, age, height_cm, weight_kg, **measurements) -> ScanResult:
    """ตรวจจับท่าทางจากภาพ RGB (numpy) แล้ววิเคราะห์ในขั้นตอนเดียว"""
    return analyze_detection(detector.detect(image_rgb), gender, age, height_cm, weight_kg, **measurements)


# ----------------------------- ภาพ: โหลดและวาดเส้นตรวจสอบ ---------------------
def load_image(file_like, max_dim: int = 960):
    """อ่านภาพจากกล้อง/อัปโหลด -> numpy RGB (หมุนตาม EXIF และย่อให้ไม่ใหญ่เกินไป) ไม่บันทึกลงดิสก์"""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(file_like)).convert("RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    return _crop_to_multiple_of_4(np.array(img))


def draw_overlay(image_rgb, detection: Optional[PoseDetection], silhouette: Optional[dict] = None):
    """
    วาดเส้นวัดบนภาพให้ผู้ใช้ตรวจสอบ คืน PIL Image
      มีผลวัดจากเงา: ฟ้า = ไหล่, เหลือง = เอว (แคบสุด), เขียว = สะโพก (กว้างสุด), ส้มโปร่งใส = บริเวณแขนที่ตัดออก
      ไม่มี: วาดแค่เส้นข้อต่อไหล่/สะโพก (ฟ้า/เขียว) ซึ่งเป็นตำแหน่งข้อต่อ ไม่ใช่ความกว้างจริง
    """
    from PIL import Image, ImageDraw

    if detection is not None:                                   # ให้ขนาดตรงกับภาพที่ใช้ตรวจจับ (ถูกตัดให้หาร 4 ลงตัว)
        image_rgb = image_rgb[: detection.image_size[1], : detection.image_size[0]]
    base = Image.fromarray(np.ascontiguousarray(image_rgb)).convert("RGBA")
    if detection is None:
        return base.convert("RGB")
    
    w, h = base.size
    lm = detection.landmarks

    if silhouette and silhouette.get("ok"):
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        r = silhouette["arm_radius_frac"] * h
        for path in silhouette["arm_paths"]:
            pts = [(x * w, y * h) for x, y in path]
            ld.line(pts, fill=(255, 140, 0, 90), width=max(1, int(2 * r)), joint="curve")
            for x, y in pts:
                ld.ellipse([x - r, y - r, x + r, y + r], fill=(255, 140, 0, 90))
        base = Image.alpha_composite(base, layer)
        draw = ImageDraw.Draw(base)
        for name, color in (("shoulder", (0, 160, 255, 255)), ("waist", (255, 215, 0, 255)), ("hip", (0, 220, 120, 255))):
            if silhouette["lines"].get(name) is None:
                continue
            y, x0, x1 = silhouette["lines"][name]
            draw.line([(x0 * w, y * h), (x1 * w, y * h)], fill=color, width=4)
            for x in (x0, x1):
                draw.ellipse([x * w - 5, y * h - 5, x * w + 5, y * h + 5], fill=color)
            draw.text((x0 * w + 8, y * h - 14), name.upper(), fill=color)
    else:
        draw = ImageDraw.Draw(base)
        pt = lambda i: (lm[i].x * w, lm[i].y * h)  # noqa: E731
        draw.line([pt(L_SHOULDER), pt(R_SHOULDER)], fill=(0, 160, 255, 255), width=4)
        draw.line([pt(L_HIP), pt(R_HIP)], fill=(0, 220, 120, 255), width=4)
        for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            x, y = pt(i)
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 80, 80, 255))
    return base.convert("RGB")