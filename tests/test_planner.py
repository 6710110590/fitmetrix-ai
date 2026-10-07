import pytest

from src.planner import (
    ENVIRONMENTS,
    EXERCISE_BANK,
    build_workout_plan,
    default_focus,
)


def make(**overrides):
    args = dict(gender="female", age=28, goal="lose_fat", environment="home", days_per_week=4)
    return build_workout_plan(**{**args, **overrides})


@pytest.mark.parametrize("days", [3, 4, 5])
def test_session_count_matches_days(days):
    assert len(make(days_per_week=days)["sessions"]) == days


def test_days_are_clamped():
    assert make(days_per_week=1)["days_per_week"] == 3
    assert make(days_per_week=9)["days_per_week"] == 5


@pytest.mark.parametrize("env", ENVIRONMENTS)
def test_exercises_come_from_selected_environment(env):
    plan = make(environment=env)
    allowed = {n for names in EXERCISE_BANK[env].values() for n in names}
    for s in plan["sessions"]:
        assert s["exercises"]
        assert all(e["name"] in allowed for e in s["exercises"])


def test_default_focus_by_gender():
    assert default_focus("female") == "hourglass"
    assert default_focus("male") == "v_shape"
    assert make()["focus"] == "hourglass"
    assert make(gender="male")["focus"] == "v_shape"


def test_every_split_group_exists_in_every_environment():
    for focus in ("hourglass", "v_shape", "balanced"):
        for d in (3, 4, 5):
            for env in ENVIRONMENTS:
                build_workout_plan("male", 30, "maintain", env, d, focus=focus)


def test_cardio_depends_on_somatotype_and_goal():
    endo = make(somatotype="endomorph")["cardio"]["sessions_per_week"]
    ecto = make(somatotype="ectomorph")["cardio"]["sessions_per_week"]
    assert endo > ecto
    assert make(goal="build_muscle")["cardio"]["sessions_per_week"] <= make(goal="lose_fat")["cardio"]["sessions_per_week"]


def test_zone2_range_uses_age():
    z = make(age=30)["cardio"]["zone2_hr_range"]
    assert z == (112, 131)   # HRmax 187 x 60-70%


def test_beginner_vs_intermediate_sets():
    assert make(level="beginner")["sessions"][0]["exercises"][0]["sets"] == "2-3"
    assert make(level="intermediate", goal="build_muscle")["sessions"][0]["exercises"][0]["sets"] == "3-4"


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        make(environment="moon")
    with pytest.raises(ValueError):
        make(goal="fly")