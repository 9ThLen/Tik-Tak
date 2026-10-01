from eval.learned_accent_calibration import half_of


def test_the_split_is_fixed_by_the_name_alone():
    assert half_of("blues.00000") == half_of("blues.00000")
    halves = [half_of(f"rock.{i:05d}") for i in range(400)]
    assert set(halves) == {"validation", "held_out"}
    # Roughly even, and decided before any result exists.
    assert 150 < halves.count("validation") < 250
