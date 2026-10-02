import numpy as np
import pytest

cv = pytest.importorskip("block_bin.conventions")

HOME_ROTVEC = [-3.1415, 0.0, 0.0016]


def test_the_guide_mapping_keys_cameras_by_their_own_names():
    assert cv.GUIDE_CAMERAS == {"top": "top", "base": "base", "wrist": "wrist"}


def test_libero_takes_two_cameras_not_three():
    """Its config.json declares camera1/2/3, but the preprocessor's rename_map fills
    only two -- the dataset has image and image2 and nothing else."""
    assert set(cv.LIBERO_CAMERAS) == {"image", "image2"}


def test_a_camera_map_is_parsed():
    assert cv.parse_camera_map("image=top,image2=wrist", {}) == {
        "image": "top",
        "image2": "wrist",
    }


def test_a_blank_map_keeps_the_convention_default():
    assert cv.parse_camera_map("  ", cv.LIBERO_CAMERAS) == cv.LIBERO_CAMERAS


def test_a_map_naming_a_camera_that_does_not_exist_is_refused():
    with pytest.raises(SystemExit, match="front"):
        cv.parse_camera_map("image=front", {})


def test_a_malformed_map_is_refused():
    with pytest.raises(SystemExit, match="policy_key=guide_camera"):
        cv.parse_camera_map("image:top", {})


def test_the_rotvec_is_moved_into_the_positive_hemisphere():
    """GUIDE's downward gripper sits at |rotvec| = pi where the sign is arbitrary;
    smolvla_libero's wx never goes negative, so an unflipped frame lands 17.8 sigma out."""
    assert cv.canonical_rotvec(HOME_ROTVEC)[0] > 0


def test_canonicalising_is_idempotent_and_leaves_positive_vectors_alone():
    once = cv.canonical_rotvec(HOME_ROTVEC)

    assert cv.canonical_rotvec(once).tolist() == once.tolist()
    assert once.tolist() == pytest.approx([3.1415, -0.0, -0.0016])


def test_the_guide_state_repeats_the_finger_and_keeps_the_sign():
    state = cv.state_for("guide", [0.1, 0.2, 0.3], HOME_ROTVEC, 0.04)

    assert state.tolist() == pytest.approx([0.1, 0.2, 0.3, -3.1415, 0.0, 0.0016, 0.04, 0.04])


def test_the_libero_state_mirrors_the_second_finger():
    """q1 spans [-0.001, +0.042] and q2 [-0.042, +0.001]; repeating one positive number
    puts q2 4.8 sigma outside."""
    state = cv.state_for("libero", [0.1, 0.2, 0.3], HOME_ROTVEC, 0.04)

    assert state[6] == pytest.approx(0.04)
    assert state[7] == pytest.approx(-0.04)
    assert state[3] > 0


def test_guide_actions_are_already_metres():
    dposition, drotvec, grip = cv.delta_for("guide", [0.01, 0, 0, 0.1, 0, 0, 0.03])

    assert dposition[0] == pytest.approx(0.01)
    assert drotvec[0] == pytest.approx(0.1)
    assert grip == pytest.approx(0.03)


def test_libero_actions_scale_position_and_rotation_differently():
    """One multiplier would put rotation an order of magnitude out."""
    dposition, drotvec, _ = cv.delta_for("libero", [1.0, 0, 0, 1.0, 0, 0, -1.0])

    assert dposition[0] == pytest.approx(cv.LIBERO_POSITION_SCALE)
    assert drotvec[0] == pytest.approx(cv.LIBERO_ROTATION_SCALE)


def test_the_libero_gripper_is_binary_and_positive_closes():
    """Its q01 is -1.0 and q99 +0.92 -- there is no absolute finger position in it."""
    assert cv.delta_for("libero", [0, 0, 0, 0, 0, 0, +1.0], gripper_open=0.04)[2] == 0.0
    assert cv.delta_for("libero", [0, 0, 0, 0, 0, 0, -1.0], gripper_open=0.04)[2] == 0.04


def test_the_operator_scale_still_applies_on_top_for_libero():
    dposition, _, _ = cv.delta_for("libero", [1.0, 0, 0, 0, 0, 0, 0], scale=0.5)

    assert dposition[0] == pytest.approx(cv.LIBERO_POSITION_SCALE * 0.5)


def test_a_wrong_width_action_names_the_likely_cause():
    with pytest.raises(ValueError, match="eef-delta-action"):
        cv.delta_for("guide", np.zeros(8))


def test_a_batched_action_is_accepted():
    dposition, _, _ = cv.delta_for("guide", np.zeros((1, 7)) + 0.1)

    assert dposition.tolist() == pytest.approx([0.1, 0.1, 0.1])


def test_the_adapted_home_pose_is_inside_what_smolvla_libero_saw():
    """The whole feasibility question for the ablation, as one assertion.

    Ranges are read off lerobot/smolvla_libero's own normaliser.
    """
    ep = pytest.importorskip("block_bin.eval_policy")
    stats = {
        "min": [-0.4828, -0.3255, 0.0081, 0.3528, -3.6414, -1.8427, -0.0014, -0.0420],
        "max": [0.2103, 0.3913, 1.3660, 3.6714, 3.5607, 1.3863, 0.0423, 0.0014],
        "mean": [-0.0465, 0.0344, 0.7646, 2.9722, -0.2205, -0.1256, 0.0269, -0.0272],
        "std": [0.1049, 0.1518, 0.3785, 0.3443, 0.9069, 0.3254, 0.0142, 0.0141],
    }
    home_position, home_rotvec = [0.0002, -0.00004, 0.4995], [-3.1415, 0.0, 0.0016]

    raw = cv.state_for("guide", home_position, home_rotvec, 0.04)
    adapted = cv.state_for("libero", home_position, home_rotvec, 0.04)

    assert [o[0] for o in ep.out_of_distribution(raw, stats)] == [3, 7]
    assert ep.out_of_distribution(adapted, stats) == []
