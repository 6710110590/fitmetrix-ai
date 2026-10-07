import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.metabolic import ProfileError
from src.scanner import (
    Landmark,
    PoseDetection,
    analyze_body,
    analyze_detection,
    assess_pose_quality,
    classify_somatotype,
    classify_visual,
    combine_evidence,
    compute_pose_metrics,
    describe_shape,
    draw_overlay,
    estimate_body_fat,
    measure_silhouette,
    scan_image,
)

W, H = 640, 960
CX = 320


def make_landmarks(vis=0.99, sh_y=(0.22, 0.22), z=(0.0, 0.0), arm_x=(0.20, 0.80)):
    """จุดข้อต่อจำลอง 33 จุด: ไหล่ y=0.22 สะโพก y=0.52 แขนห้อยแนวตั้งที่ x = arm_x"""
    lm = [Landmark(0.5, 0.5, 0.0, vis) for _ in range(33)]
    lm[11], lm[12] = Landmark(0.36, sh_y[0], z[0], vis), Landmark(0.64, sh_y[1], z[1], vis)
    lm[13], lm[14] = Landmark(arm_x[0], 0.40, 0.0, vis), Landmark(arm_x[1], 0.40, 0.0, vis)
    lm[15], lm[16] = Landmark(arm_x[0], 0.54, 0.0, vis), Landmark(arm_x[1], 0.54, 0.0, vis)
    for i in (17, 19, 21):
        lm[i] = Landmark(arm_x[0], 0.58, 0.0, vis)
    for i in (18, 20, 22):
        lm[i] = Landmark(arm_x[1], 0.58, 0.0, vis)
    lm[23], lm[24] = Landmark(0.44, 0.52, 0.0, vis), Landmark(0.56, 0.52, 0.0, vis)
    lm[25], lm[26] = Landmark(0.44, 0.75, 0.0, vis), Landmark(0.56, 0.75, 0.0, vis)
    lm[27], lm[28] = Landmark(0.44, 0.95, 0.0, vis), Landmark(0.56, 0.95, 0.0, vis)
    return lm


def make_mask(shoulder=220, waist=160, hip=210, arm_gap=20, arm_w=40, top=60, bottom=930):
    """เงาจำลองขนาด 640x960 ความกว้างลำตัว (พิกเซล) ที่ระดับไหล่/เอว/สะโพกตามพารามิเตอร์
    ไหล่ y=211 เอวแคบสุด y=400 สะโพกกว้างสุด y=520 ขาสองข้างมีช่องว่างตรงกลาง
    arm_gap = ระยะห่างระหว่างแขนกับลำตัว (0 = แขนแนบลำตัว, None = ไม่มีแขน)"""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    d.ellipse([CX - 45, top, CX + 45, top + 100], fill=255)                            # ศีรษะ
    d.polygon([(CX - shoulder / 2, 211), (CX + shoulder / 2, 211),
               (CX + (shoulder + waist) / 4, 300), (CX + waist / 2, 400),
               (CX + (waist + hip) / 2 / 2 + 5, 470), (CX + hip / 2, 520), (CX + hip / 2 - 8, 580),
               (CX - hip / 2 + 8, 580), (CX - hip / 2, 520), (CX - (waist + hip) / 2 / 2 - 5, 470),
               (CX - waist / 2, 400), (CX - (shoulder + waist) / 4, 300)], fill=255)   # ลำตัว
    for sx in (-1, 1):                                                                 # ขา
        x0 = CX + sx * 6 if sx > 0 else CX - 6 - 70
        d.rectangle([x0, 570, x0 + 70, bottom], fill=255)
    if arm_gap is not None:                                                            # แขน
        for sx in (-1, 1):
            edge = CX + sx * (hip / 2 + arm_gap)
            x0, x1 = (edge, edge + arm_w) if sx > 0 else (edge - arm_w, edge)
            d.rectangle([x0, 215, x1, 565], fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0


def arm_x_for(hip=210, arm_gap=20, arm_w=40):
    c = (hip / 2 + arm_gap + arm_w / 2) / W
    return (0.5 - c, 0.5 + c)


def detection(mask="default", **kw):
    lm = make_landmarks(arm_x=arm_x_for()) if "lm" not in kw else kw.pop("lm")
    m = make_mask() if isinstance(mask, str) else mask
    return PoseDetection(landmarks=lm, image_size=(W, H), mask=m)


class FakeDetector:
    def __init__(self, det):
        self.det = det

    def detect(self, image):
        return self.det


# ---------- คุณภาพท่าทาง ----------
def test_tilt_is_detected():
    lm = make_landmarks(sh_y=(0.22, 0.28))
    m = compute_pose_metrics(lm, W, H)
    assert m["shoulder_tilt_deg"] > 8
    _, warnings = assess_pose_quality(lm, m)
    assert any("เอียง" in w for w in warnings)


def test_low_visibility_is_blocking_error():
    lm = make_landmarks()
    lm[11] = Landmark(0.36, 0.22, 0.0, 0.2)
    errors, _ = assess_pose_quality(lm, compute_pose_metrics(lm, W, H))
    assert errors


def test_missing_legs_is_only_warning():
    lm = make_landmarks()
    lm[27] = Landmark(0.44, 1.2, 0.0, 0.1)
    errors, warnings = assess_pose_quality(lm, compute_pose_metrics(lm, W, H))
    assert not errors and any("เต็มตัว" in w for w in warnings)


def test_turned_body_warns():
    lm = make_landmarks(z=(-0.2, 0.2))
    _, warnings = assess_pose_quality(lm, compute_pose_metrics(lm, W, H))
    assert any("หันตรง" in w for w in warnings)


def test_clean_pose_has_no_issues():
    lm = make_landmarks()
    assert assess_pose_quality(lm, compute_pose_metrics(lm, W, H)) == ([], [])


# ---------- วัดเงาร่างกาย ----------
def test_silhouette_widths_with_separate_arms():
    s = measure_silhouette(make_landmarks(arm_x=arm_x_for()), make_mask())
    assert s["ok"], s["problems"]
    assert s["shoulder_width_px"] == pytest.approx(220, abs=8)
    assert s["waist_width_px"] == pytest.approx(160, abs=8)
    assert s["hip_width_px"] == pytest.approx(210, abs=8)
    assert s["waist_hip_ratio"] == pytest.approx(160 / 210, abs=0.04)
    assert s["arms_removed"] == 2


def test_hip_is_measured_from_silhouette_not_from_joints():
    # จุดข้อต่อสะโพกห่างกันแค่ 0.12*640 = 77 px แต่สะโพกจริงกว้าง 210 px
    s = measure_silhouette(make_landmarks(arm_x=arm_x_for()), make_mask())
    joint_width = (0.56 - 0.44) * W
    assert s["hip_width_px"] > 2 * joint_width


def test_arm_removal_fixes_touching_arms():
    """แขนแนบลำตัว: ถ้าไม่ตัดแขน เอว/สะโพกจะกว้างเกินจริงมาก ตัดแล้วต้องใกล้ค่าจริงกว่า"""
    mask = make_mask(arm_gap=0)
    lm = make_landmarks(arm_x=arm_x_for(arm_gap=0))
    s = measure_silhouette(lm, mask)
    assert s["ok"], s["problems"]
    assert s["arms_removed"] == 2
    naive_waist = 160 + 2 * 40
    assert abs(s["waist_width_px"] - 160) < abs(naive_waist - 160)
    assert abs(s["waist_width_px"] - 160) / 160 < 0.2
    assert abs(s["hip_width_px"] - 210) / 210 < 0.2


def test_missing_arm_landmarks_warn_but_still_measure():
    lm = make_landmarks(arm_x=arm_x_for())
    lm[13] = Landmark(0.2, 0.4, 0.0, 0.1)
    s = measure_silhouette(lm, make_mask())
    assert s["ok"] and s["arms_removed"] == 1
    assert any("มองไม่เห็นแขน" in p for p in s["problems"])


def test_empty_mask_not_ok():
    assert not measure_silhouette(make_landmarks(), np.zeros((H, W), dtype=np.float32))["ok"]


def test_body_cut_at_edge_not_ok():
    s = measure_silhouette(make_landmarks(arm_x=arm_x_for()), make_mask(bottom=959))
    assert not s["ok"] and any("ถูกตัดขอบ" in p for p in s["problems"])


def test_tiny_body_not_ok():
    small = np.zeros((H, W), dtype=np.float32)
    small[400:600, 300:340] = 1.0
    assert not measure_silhouette(make_landmarks(), small)["ok"]


def test_lines_are_normalized_and_ordered():
    s = measure_silhouette(make_landmarks(arm_x=arm_x_for()), make_mask())
    (ys, _, _), (yw, _, _), (yh, x0, x1) = s["lines"]["shoulder"], s["lines"]["waist"], s["lines"]["hip"]
    assert 0 < ys < yw < yh < 1 and 0 <= x0 < x1 <= 1


def test_describe_shape():
    assert describe_shape("male", 1.45) == "v_taper"
    assert describe_shape("male", 1.2) == "balanced"
    assert describe_shape("male", 1.0) == "pear"
    assert describe_shape("female", 1.25) == "v_taper"
    assert describe_shape("female", 1.05) == "balanced"
    assert describe_shape("female", 0.9) == "pear"


# ---------- ไขมันในร่างกาย ----------
def test_navy_male_hand_calc():
    bf, method = estimate_body_fat("male", 30, 175, 75, waist_cm=85, neck_cm=38)
    assert method == "navy" and bf == pytest.approx(17.0, abs=0.3)


def test_navy_female_needs_hip():
    _, method = estimate_body_fat("female", 30, 165, 60, waist_cm=75, neck_cm=32)
    assert method == "deurenberg"
    _, method = estimate_body_fat("female", 30, 165, 60, waist_cm=75, neck_cm=32, hip_cm=98)
    assert method == "navy"


def test_deurenberg_hand_calc():
    # BMI 72/1.75^2 = 23.51 -> 1.2*23.51 + 0.23*28 - 10.8 - 5.4 = 18.46
    bf, method = estimate_body_fat("male", 28, 175, 72)
    assert method == "deurenberg" and bf == pytest.approx(18.5, abs=0.1)


def test_invalid_navy_falls_back():
    _, method = estimate_body_fat("male", 30, 175, 75, waist_cm=30, neck_cm=38)
    assert method == "deurenberg"


# ---------- จำแนก ----------
@pytest.mark.parametrize(
    "gender, height, weight, expected",
    [("male", 180, 60, "ectomorph"), ("male", 175, 72, "mesomorph"), ("male", 170, 95, "endomorph"),
     ("female", 165, 50, "ectomorph"), ("female", 165, 60, "mesomorph"), ("female", 160, 85, "endomorph")],
)
def test_numeric_classification(gender, height, weight, expected):
    assert analyze_body(gender, 28, height, weight).somatotype == expected


@pytest.mark.parametrize(
    "gender, waist, hip, expected",
    [("female", 0.252, 0.282, "endomorph"),   # สัดส่วนจากภาพตัวอย่างที่ผู้ใช้ส่งมา (วัดโดยประมาณ)
     ("female", 0.170, 0.220, "mesomorph"), ("female", 0.140, 0.190, "ectomorph"),
     ("male", 0.210, 0.200, "endomorph"), ("male", 0.175, 0.197, "mesomorph"), ("male", 0.140, 0.170, "ectomorph"),
     ("female", 0.160, 0.265, "endomorph")],    # สะโพกกว้างมากแม้เอวไม่กว้าง
)
def test_visual_classification(gender, waist, hip, expected):
    assert classify_visual(gender, waist, hip) == expected


def test_combine_agree_and_numbers_only():
    assert combine_evidence("mesomorph", "mesomorph", False)[:2] == ("mesomorph", "agree")
    assert combine_evidence("mesomorph", None, False)[:2] == ("mesomorph", "numbers_only")


def test_combine_conflict_prefers_visual_without_measured_fat():
    final, agreement, reasons = combine_evidence("mesomorph", "endomorph", measured_body_fat=False)
    assert (final, agreement) == ("endomorph", "conflict") and reasons


def test_combine_conflict_prefers_numbers_when_measured():
    assert combine_evidence("mesomorph", "endomorph", measured_body_fat=True)[0] == "mesomorph"


def test_combine_big_conflict_falls_to_mesomorph():
    assert combine_evidence("ectomorph", "endomorph", False)[0] == "mesomorph"
    assert combine_evidence("endomorph", "ectomorph", True)[0] == "mesomorph"


# ---------- analyze_body ----------
def test_without_photo_is_low_confidence():
    r = analyze_body("male", 28, 175, 72)
    assert not r.photo_used and r.confidence == "low" and r.agreement == "numbers_only"


def test_full_figure_overrides_average_numbers():
    """ตัวเลขเฉลี่ย (Mesomorph) แต่เงาอวบชัดเจน -> สรุป Endomorph เพราะไม่มีรอบเอว/คอที่วัดจริง"""
    full = make_mask(shoulder=240, waist=210, hip=235)
    r = analyze_body("male", 28, 175, 72, detection=detection(mask=full, lm=make_landmarks(arm_x=arm_x_for(hip=235))))
    assert r.numeric_somatotype == "mesomorph" and r.visual_somatotype == "endomorph"
    assert r.somatotype == "endomorph" and r.agreement == "conflict" and r.confidence == "low"
    assert r.photo_used and r.silhouette["ok"]


def test_measured_body_fat_wins_over_silhouette():
    full = make_mask(shoulder=240, waist=210, hip=235)
    r = analyze_body("male", 28, 175, 72, waist_cm=85, neck_cm=38,
                     detection=detection(mask=full, lm=make_landmarks(arm_x=arm_x_for(hip=235))))
    assert r.body_fat_method == "navy" and r.somatotype == "mesomorph" and r.agreement == "conflict"


def test_agreement_gives_medium_confidence():
    r = analyze_body("female", 28, 165, 60, detection=detection(mask=make_mask(shoulder=200, waist=150, hip=170)))
    assert r.agreement == "agree" and r.confidence == "medium"


def test_no_mask_falls_back_to_numbers():
    det = PoseDetection(landmarks=make_landmarks(), image_size=(W, H), mask=None)
    r = analyze_body("male", 28, 175, 72, detection=det)
    assert not r.photo_used and any("เงาร่างกาย" in w for w in r.warnings)


def test_bad_pose_falls_back_to_numbers():
    det = PoseDetection(landmarks=make_landmarks(vis=0.1), image_size=(W, H), mask=make_mask())
    r = analyze_body("male", 28, 175, 72, detection=det)
    assert not r.photo_used and r.photo_errors and r.somatotype


def test_cut_off_body_is_reported_as_photo_error():
    det = PoseDetection(landmarks=make_landmarks(arm_x=arm_x_for()), image_size=(W, H), mask=make_mask(bottom=959))
    r = analyze_body("male", 28, 175, 72, detection=det)
    assert not r.photo_used and any("ถูกตัดขอบ" in e for e in r.photo_errors)


def test_scan_image_no_person():
    r = scan_image(np.zeros((H, W, 3), dtype=np.uint8), FakeDetector(None), "male", 28, 175, 72)
    assert not r.photo_used and any("ไม่พบคน" in e for e in r.photo_errors)


def test_scan_image_with_person():
    r = scan_image(np.zeros((H, W, 3), dtype=np.uint8), FakeDetector(detection()), "female", 28, 165, 55)
    assert r.photo_used and r.silhouette["hip_width_px"] > 150


def test_invalid_profile_rejected():
    with pytest.raises(ProfileError):
        analyze_body("male", 16, 175, 72)
    with pytest.raises(ProfileError):
        analyze_body("x", 28, 175, 72)


def test_draw_overlay_variants_keep_size():
    r = analyze_detection(detection(), "female", 28, 165, 60)
    img_blank = np.zeros((H, W, 3), dtype=np.uint8)
    assert draw_overlay(img_blank, detection(), r.silhouette).size == (W, H)          # มีเส้นวัดจากเงา
    assert draw_overlay(img_blank, detection(), None).size == (W, H)                  # มีแต่เส้นข้อต่อ
    assert draw_overlay(img_blank, None, None).size == (W, H)                         # ไม่พบคน


def test_to_dict():
    keys = analyze_body("male", 28, 175, 72).to_dict().keys()
    assert {"somatotype", "confidence", "agreement", "numeric_somatotype", "visual_somatotype", "reasons"} <= keys

# ---------- กันโปรแกรมดับจากภาพที่ความกว้างไม่หาร 4 ลงตัว ----------
import gc
import io

from src.scanner import _crop_to_multiple_of_4, _mask_from_result, load_image


class FakeResult:
    def __init__(self, masks):
        self.segmentation_masks = masks


def test_crop_to_multiple_of_4():
    for w, expected in [(358, 356), (357, 356), (355, 352), (360, 360), (4, 4)]:
        out = _crop_to_multiple_of_4(np.zeros((10, w, 3), dtype=np.uint8))
        assert out.shape[1] == expected and out.flags["C_CONTIGUOUS"]


def test_load_image_width_is_multiple_of_4():
    buf = io.BytesIO()
    Image.new("RGB", (357, 441), (200, 100, 50)).save(buf, format="PNG")
    buf.seek(0)
    assert load_image(buf).shape == (441, 356, 3)


def test_mask_with_odd_width_is_skipped_not_crashed():
    """เดิม numpy_view() บนภาพ float กว้าง 53/357/358 พิกเซลทำให้ process ดับทั้งตัว (abort) ตอนนี้ต้องข้ามอย่างปลอดภัย"""
    mp = pytest.importorskip("mediapipe")
    for w in (53, 357, 358):
        img = mp.Image(image_format=mp.ImageFormat.VEC32F1, data=np.random.rand(40, w).astype(np.float32))
        assert _mask_from_result(FakeResult([img])) is None


def test_mask_is_copied_and_survives_after_image_is_freed():
    mp = pytest.importorskip("mediapipe")
    data = np.random.rand(40, 52).astype(np.float32)
    img = mp.Image(image_format=mp.ImageFormat.VEC32F1, data=data)
    mask = _mask_from_result(FakeResult([img]))
    del img
    gc.collect()
    assert mask.shape == (40, 52) and np.allclose(mask, data)


def test_mask_missing_returns_none():
    assert _mask_from_result(FakeResult(None)) is None
    assert _mask_from_result(FakeResult([])) is None


def test_overlay_matches_detection_size_when_image_is_wider():
    det = PoseDetection(landmarks=make_landmarks(arm_x=arm_x_for()), image_size=(W, H), mask=make_mask())
    r = analyze_detection(det, "female", 28, 165, 60)
    wider = np.zeros((H, W + 3, 3), dtype=np.uint8)           # ภาพต้นฉบับกว้างกว่าที่ใช้ตรวจจับ 3 พิกเซล
    assert draw_overlay(wider, det, r.silhouette).size == (W, H)


def test_detect_integration_with_odd_width_image_using_fake_landmarker():
    """ส่งภาพกว้างคี่เข้า PoseDetector.detect จริง (landmarker จำลอง) ต้องไม่ดับ และได้เงาขนาดตรงกับภาพที่ถูกตัด"""
    mp = pytest.importorskip("mediapipe")
    from src.scanner import PoseDetector

    class FakePoint:
        x, y, z, visibility = 0.5, 0.5, 0.0, 0.9

    class FakeLandmarker:
        def detect(self, mp_image):
            mask = mp.Image(image_format=mp.ImageFormat.VEC32F1,
                            data=np.ones((mp_image.height, mp_image.width), dtype=np.float32))

            class Result:
                pose_landmarks = [[FakePoint() for _ in range(33)]]
                segmentation_masks = [mask]

            return Result()

    detector = object.__new__(PoseDetector)       # ข้าม __init__ (ไม่ต้องใช้ไฟล์โมเดล)
    detector._mp, detector._landmarker = mp, FakeLandmarker()
    det = detector.detect(np.zeros((441, 357, 3), dtype=np.uint8))
    assert det.image_size == (356, 441)
    assert det.mask.shape == (441, 356) and float(det.mask.min()) == 1.0

    # ---------- ศีรษะ/เท้าชิดขอบ และแขนที่ยกสูง/มองไม่เห็น ----------
def _with_raised_arm(mask, x0=110, x1=150, y0=0, y1=300):
    """เพิ่มแขนที่ยกสูงจนปลายมือชิดขอบบนของภาพ (อยู่ห่างศีรษะ)"""
    out = mask.copy()
    out[y0:y1, x0:x1] = 1.0
    return out


def test_raised_hand_at_top_edge_is_not_a_head_cut():
    s = measure_silhouette(make_landmarks(arm_x=arm_x_for()), _with_raised_arm(make_mask()))
    assert s["ok"], s["problems"]


def test_head_touching_top_edge_reports_head_only():
    s = measure_silhouette(make_landmarks(arm_x=arm_x_for()), make_mask(top=0))
    assert not s["ok"]
    assert any("ศีรษะ" in p for p in s["problems"]) and not any("เท้า" in p for p in s["problems"])


def test_feet_touching_bottom_edge_reports_feet_only():
    s = measure_silhouette(make_landmarks(arm_x=arm_x_for()), make_mask(bottom=959))
    assert not s["ok"]
    assert any("เท้า" in p for p in s["problems"]) and not any("ศีรษะ" in p for p in s["problems"])


def test_head_and_raised_hand_both_at_edge_still_rejected():
    s = measure_silhouette(make_landmarks(arm_x=arm_x_for()), _with_raised_arm(make_mask(top=0)))
    assert not s["ok"] and any("ศีรษะ" in p for p in s["problems"])


def test_hidden_nose_falls_back_to_global_top():
    lm = make_landmarks(arm_x=arm_x_for())
    lm[0] = Landmark(0.5, 0.1, 0.0, 0.1)
    s = measure_silhouette(lm, _with_raised_arm(make_mask()))
    assert not s["ok"] and any("ศีรษะ" in p for p in s["problems"])    # ไม่มีจมูกบอกตำแหน่ง ใช้ยอดสูงสุดของเงา (มือ) ตามปลอดภัยไว้ก่อน


def test_missing_arm_blocks_endomorph_from_image():
    """แขนหนึ่งข้างมองไม่เห็น: เงาอวบอาจเกิดจากแขนติดลำตัว จึงไม่เชื่อภาพที่ชี้ว่า Endomorph (ใช้ตัวเลขแทน)"""
    full = make_mask(shoulder=240, waist=210, hip=235)
    lm = make_landmarks(arm_x=arm_x_for(hip=235))
    lm[13] = Landmark(0.2, 0.4, 0.0, 0.1)
    r = analyze_body("male", 28, 175, 72, detection=detection(mask=full, lm=lm))
    assert r.visual_somatotype is None and not r.photo_used
    assert r.somatotype == r.numeric_somatotype == "mesomorph"
    assert r.silhouette.get("ok") and any("ไม่ใช้ภาพนี้ตัดสินว่า Endomorph" in w for w in r.warnings)


def test_missing_arm_still_allows_lean_or_average_from_image():
    lm = make_landmarks(arm_x=arm_x_for())
    lm[13] = Landmark(0.2, 0.4, 0.0, 0.1)
    r = analyze_body("female", 28, 165, 60, detection=detection(mask=make_mask(shoulder=200, waist=150, hip=170), lm=lm))
    assert r.photo_used and r.visual_somatotype == "mesomorph"

# ---------- ไหล่ที่มีแขนยกสูงติดเงา / มือวางหน้าลำตัว ----------
def _mask_with_elbow_at_shoulder(extra=130):
    """แขนที่งอยกสูง ศอกยื่นออกมาที่ระดับไหล่ซ้าย ติดกับลำตัว ทำให้เงาตรงแถวไหล่กว้างเกินจริง"""
    m = make_mask()
    m[205:235, int(CX - 110 - extra):int(CX - 110)] = 1.0
    return m


def test_shoulder_dropped_when_elbow_sticks_out_at_shoulder_level():
    s = measure_silhouette(make_landmarks(arm_x=arm_x_for()), _mask_with_elbow_at_shoulder())
    assert s["ok"]                                        # เอว/สะโพกยังวัดได้
    assert s["shoulder_width_px"] is None and s["shoulder_ratio"] is None and s["shoulder_hip_ratio"] is None
    assert s["lines"]["shoulder"] is None and s["lines"]["waist"] and s["lines"]["hip"]
    assert any("วัดความกว้างไหล่ไม่ได้" in p for p in s["problems"])


def test_normal_shoulder_is_kept():
    s = measure_silhouette(make_landmarks(arm_x=arm_x_for()), make_mask())
    assert s["shoulder_width_px"] and s["lines"]["shoulder"]


def test_analyze_body_still_classifies_when_shoulder_dropped():
    det = detection(mask=_mask_with_elbow_at_shoulder())
    r = analyze_body("female", 28, 165, 60, detection=det)
    assert r.photo_used and r.shape is None and r.shape_label is None
    assert any("วัดความกว้างไหล่ไม่ได้" in w for w in r.warnings)
    assert "เอว" in " ".join(r.reasons)


def _landmarks_with_wrist(x, y, which="right"):
    lm = make_landmarks(arm_x=arm_x_for())
    wrist, hand = (16, (18, 20, 22)) if which == "right" else (15, (17, 19, 21))
    lm[wrist] = Landmark(x, y, 0.0, 0.99)
    for i in hand:
        lm[i] = Landmark(x, y, 0.0, 0.99)
    return lm


def test_hand_in_front_of_waist_is_rejected():
    s = measure_silhouette(_landmarks_with_wrist(0.52, 0.42), make_mask())
    assert not s["ok"] and any("หน้าลำตัว" in p for p in s["problems"])


def test_hand_in_front_of_hip_is_rejected():
    s = measure_silhouette(_landmarks_with_wrist(0.47, 0.54, which="left"), make_mask())
    assert not s["ok"] and any("หน้าลำตัว" in p for p in s["problems"])


def test_hand_raised_to_head_is_not_in_front_of_torso():
    """มือยกขึ้นที่ศีรษะ (สูงกว่าไหล่) ไม่บังช่วงเอว/สะโพก ต้องวัดได้"""
    s = measure_silhouette(_landmarks_with_wrist(0.46, 0.10), make_mask())
    assert s["ok"], s["problems"]


def test_hand_hanging_beside_body_is_fine():
    s = measure_silhouette(_landmarks_with_wrist(0.78, 0.45), make_mask())
    assert s["ok"], s["problems"]


def test_hand_on_waist_photo_falls_back_to_numbers():
    det = detection(lm=_landmarks_with_wrist(0.52, 0.42))
    r = analyze_body("female", 28, 165, 60, detection=det)
    assert not r.photo_used and any("หน้าลำตัว" in e for e in r.photo_errors)


def test_overlay_skips_missing_shoulder_line():
    det = detection(mask=_mask_with_elbow_at_shoulder())
    r = analyze_body("female", 28, 165, 60, detection=det)
    assert draw_overlay(np.zeros((H, W, 3), dtype=np.uint8), det, r.silhouette).size == (W, H)