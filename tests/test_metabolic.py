import pytest

from src.metabolic import (
    ProfileError,
    build_nutrition_plan,
    calc_tdee,
)

MALE = dict(gender="male", age=30, height_cm=180, weight_kg=80, target_weight_kg=75,
            activity_level="moderate")


def plan(**overrides):
    args = {**MALE, "goal": "lose_fat", **overrides}
    return build_nutrition_plan(**args)


# ---------- BMR / TDEE (คำนวณมือ) ----------
def test_bmr_male_hand_calc():
    # 10*80 + 6.25*180 - 5*30 + 5 = 1780
    assert plan().bmr == 1780


def test_bmr_female_hand_calc():
    # 10*60 + 6.25*165 - 5*28 - 161 = 1330.25
    p = plan(gender="female", height_cm=165, weight_kg=60, age=28, target_weight_kg=57)
    assert p.bmr == 1330


def test_tdee_factor():
    assert calc_tdee(1780, "moderate") == pytest.approx(2759.0)
    with pytest.raises(ValueError):
        calc_tdee(1780, "extreme")


# ---------- เป้าหมายแคลอรี ----------
def test_lose_fat_default_15_percent():
    p = plan()
    assert p.tdee == 2759
    assert p.target_calories == pytest.approx(2759 * 0.85, abs=1)
    assert p.calorie_adjustment < 0 and p.weekly_weight_change_kg < 0


def test_build_muscle_surplus_10_percent():
    p = plan(goal="build_muscle", target_weight_kg=85)
    assert p.target_calories == pytest.approx(2759 * 1.10, abs=1)
    assert p.weekly_weight_change_kg > 0


def test_maintain_equals_tdee():
    p = plan(goal="maintain", target_weight_kg=80)
    assert p.target_calories == p.tdee and not p.warnings


def test_adjust_pct_is_clamped():
    p = plan(adjust_pct=0.5)
    assert any("ปรับสัดส่วนแคลอรี" in w for w in p.warnings)


# ---------- กฎความปลอดภัย ----------
def test_calorie_floor_applied():
    p = plan(gender="female", age=40, height_cm=150, weight_kg=45, target_weight_kg=43,
             activity_level="sedentary")
    assert p.target_calories >= 1200 and p.target_calories >= p.bmr
    assert any("ไม่ควรต่ำกว่า" in w for w in p.warnings)


def test_weekly_loss_capped_at_one_percent():
    p = plan(gender="female", age=25, height_cm=160, weight_kg=50, target_weight_kg=48,
             activity_level="very_active", adjust_pct=0.25)
    # 1% x 50 kg x 7700 / 7 = 550 kcal/วัน
    assert p.calorie_adjustment == -550
    assert p.weekly_weight_change_kg >= -0.5 - 1e-9


def test_underweight_gets_no_deficit():
    p = plan(age=20, height_cm=170, weight_kg=50, target_weight_kg=48)
    assert p.target_calories == p.tdee
    assert any("BMI" in w for w in p.warnings)


def test_minor_rejected():
    with pytest.raises(ProfileError):
        plan(age=16)


def test_invalid_inputs_collect_errors():
    with pytest.raises(ProfileError) as e:
        plan(gender="x", goal="fly", activity_level="none")
    assert len(e.value.errors) == 3


# ---------- ระยะเวลาถึงเป้าหมาย ----------
def test_weeks_to_goal():
    p = plan()
    # ลด 5 กก. ที่ ~0.376 กก./สัปดาห์ ~ 13.3 สัปดาห์
    assert p.weeks_to_goal == pytest.approx(13.3, abs=0.2)


def test_opposite_direction_target_warns():
    p = plan(target_weight_kg=85)
    assert p.weeks_to_goal is None
    assert any("สวนทาง" in w for w in p.warnings)


# ---------- Macros ----------
def test_macros_add_up_to_target():
    p = plan()
    total = sum(p.macros[k]["kcal"] for k in ("protein", "carbs", "fat"))
    assert total == pytest.approx(p.target_calories, rel=0.02)
    assert sum(p.macros[k]["pct"] for k in ("protein", "carbs", "fat")) == pytest.approx(100, abs=0.2)


def test_somatotype_macro_differences():
    ecto = plan(somatotype="ectomorph").macros
    endo = plan(somatotype="endomorph").macros
    assert endo["protein_g_per_kg"] > ecto["protein_g_per_kg"]
    assert ecto["carbs"]["grams"] > endo["carbs"]["grams"]


def test_to_dict():
    assert {"bmr", "tdee", "target_calories", "macros", "warnings", "notes"} <= plan().to_dict().keys()