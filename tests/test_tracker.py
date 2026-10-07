import pytest

from src.tracker import (
    CalorieTracker,
    UserProfile,
    ValidationError,
    Workout,
    calc_hr_max,
    classify_zone,
    keytel_calories,
    zone_table,
)

tracker = CalorieTracker(policy="fallback")
tracker_warn = CalorieTracker(policy="model_warn")

MALE = UserProfile(gender="male", age=28, height_cm=175, weight_kg=72)
FEMALE = UserProfile(gender="female", age=35, height_cm=162, weight_kg=58)


# ---------- Zones ----------
def test_hr_max_tanaka():
    assert calc_hr_max(30) == pytest.approx(187.0)


@pytest.mark.parametrize(
    "hr, zone",
    [(80, 0), (95, 1), (115, 2), (140, 3), (160, 4), (175, 5), (200, 5)],
)
def test_zone_boundaries_age_30(hr, zone):
    # HRmax = 187 -> Z1 93.5, Z2 112.2, Z3 131.0, Z4 149.6, Z5 168.3
    assert classify_zone(hr, 30)["zone"] == zone


def test_zone_table_has_five_rows():
    assert len(zone_table(30)) == 5


# ---------- สูตร Keytel ----------
def test_keytel_male_hand_calc():
    # (-55.0969 + 0.6309*150 + 0.1988*80 + 0.2017*30) / 4.184 * 30 = 440.9
    assert keytel_calories("male", 30, 80, 150, 30) == pytest.approx(440.9, abs=0.5)


def test_keytel_never_negative():
    assert keytel_calories("female", 25, 60, 50, 10) == 0.0


# ---------- Validation ----------
def test_invalid_gender():
    with pytest.raises(ValidationError):
        tracker.predict(UserProfile("other", 28, 175, 72), Workout(20, 110))


def test_impossible_values_collect_all_errors():
    with pytest.raises(ValidationError) as e:
        tracker.predict(UserProfile("male", 28, 175, 72), Workout(duration_min=0, heart_rate=300))
    assert len(e.value.errors) == 2


def test_non_numeric_input():
    with pytest.raises(ValidationError):
        tracker.predict(UserProfile("male", "abc", 175, 72), Workout(20, 110))


def test_gender_aliases_and_case():
    r = tracker.predict(UserProfile(" Male ", 28, 175, 72), Workout(20, 110))
    assert r.method == "ml_model"


# ---------- การทำนาย ----------
def test_in_range_uses_model():
    r = tracker.predict(MALE, Workout(duration_min=20, heart_rate=110))
    assert r.method == "ml_model" and r.confidence == "high"
    assert r.calories > 0 and not r.out_of_range


def test_more_duration_burns_more():
    short = tracker.predict(MALE, Workout(10, 110)).calories
    long_ = tracker.predict(MALE, Workout(25, 110)).calories
    assert long_ > short


def test_out_of_range_fallback_to_formula():
    r = tracker.predict(MALE, Workout(duration_min=60, heart_rate=150))
    assert r.method == "keytel_formula" and r.confidence == "medium"
    assert "Duration" in r.out_of_range and "Heart_Rate" in r.out_of_range
    assert r.warnings and r.calories > 0


def test_out_of_range_model_warn_policy():
    r = tracker_warn.predict(MALE, Workout(duration_min=60, heart_rate=150))
    assert r.method == "ml_model" and r.confidence == "low" and r.warnings


def test_hr_above_hrmax_warns():
    r = tracker.predict(UserProfile("male", 60, 175, 72), Workout(20, 190))
    assert any("HRmax" in w for w in r.warnings)


def test_strength_activity_warns():
    r = tracker.predict(FEMALE, Workout(20, 110, activity="strength"))
    assert any("เวท" in w for w in r.warnings)


def test_result_to_dict():
    d = tracker.predict(MALE, Workout(20, 110)).to_dict()
    assert {"calories", "method", "confidence", "zone", "hr_max", "warnings"} <= d.keys()