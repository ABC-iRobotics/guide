import numpy as np
import pytest

# Pulls in lerobot + the ROS workspace through eval_policy_pink, i.e. it only imports
# under the Isaac venv interpreter with the workspace sourced. Skip elsewhere.
pg = pytest.importorskip("block_bin.probe_grounding")


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("0-3", [0, 1, 2, 3]),
        ("0,3,7", [0, 3, 7]),
        ("0-2,10", [0, 1, 2, 10]),
        ("5", [5]),
        (" 1 , 2 ", [1, 2]),
        ("4-4", [4]),
    ],
)
def test_seed_specs(spec, expected):
    assert pg.parse_seeds(spec) == expected


def test_seeds_keep_the_order_they_were_given():
    """The pairs run in this order, and a probe stopped half way should have run the
    seeds the caller listed first."""
    assert pg.parse_seeds("7,1,4") == [7, 1, 4]


def test_a_backwards_range_is_an_error_not_an_empty_run():
    with pytest.raises(SystemExit, match="backwards"):
        pg.parse_seeds("9-0")


def test_no_seeds_is_an_error():
    with pytest.raises(SystemExit, match="empty"):
        pg.parse_seeds(" , ")


def test_the_instruction_matches_the_scene_template():
    """Scene.randomize_preprocess builds exactly this sentence."""
    assert pg.instruction_for("blue", "left") == "Put the blue block in the left bin."


def test_bin_zero_is_left():
    """Scene.is_success_preprocess grades bin_{0 if left else 1}; the probe must agree
    or every verdict is mirrored."""
    assert pg.SIDES["left"] == 0
    assert pg.SIDES["right"] == 1


def test_following_both_instructions_is_grounded():
    """Told left it went left, told right it went right. The pair is symmetric, so no
    verdict needs to know which side the scene drew."""
    assert pg.verdict("left", "right") == "followed"


def test_the_same_bin_twice_is_the_words_being_ignored():
    """The signature of a memorised routine: the scene decided, the sentence did not."""
    assert pg.verdict("right", "right") == "ignored"
    assert pg.verdict("left", "left") == "ignored"


def test_a_bin_bias_cannot_score_as_grounded():
    """A policy that always picks the right bin gets one trial of each pair correct,
    which is exactly why the pair -- not the trial -- is the unit."""
    assert pg.verdict("right", "right") == "ignored"


def test_no_placement_is_not_evidence_either_way():
    assert pg.verdict("neither", "right") == "incomplete"
    assert pg.verdict("left", "neither") == "incomplete"
    assert pg.verdict("neither", "neither") == "incomplete"


def test_both_bins_backwards_is_named_not_hidden():
    assert pg.verdict("right", "left") == "inverted"


class FakeResponse:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class FakePose:
    def __init__(self, x, y, z=1.0):
        self.pose = FakeResponse(position=FakeResponse(x=x, y=y, z=z))


class FakeSimRobot:
    """Answers CollisionRequest and PoseRequest for one placement of the block."""

    def __init__(self, overlaps, block=(0.25, -0.40), bins=None):
        self.overlaps = overlaps
        self.block = block
        self.bins = bins or {"bin_0": (0.25, -0.40), "bin_1": (0.25, 0.52)}
        self.collision = object()
        self.pose = object()

    def callService(self, client, request):
        if client is self.collision:
            name = request.prim2.rsplit("/", 1)[-1]
            return FakeResponse(collision=self.overlaps.get(name, False))
        name = request.path.rsplit("/", 1)[-1]
        x, y = self.bins[name] if name in self.bins else self.block
        return FakePose(x, y)


def where(overlaps, block=(0.25, -0.40)):
    robot = FakeSimRobot(overlaps, block=block)
    return pg.where_is(robot, 0, "/blocks/blue_block")


def test_a_block_overlapping_one_bin_is_in_that_bin():
    assert where({"bin_0": True})["reached"] == "left"
    assert where({"bin_1": True})["reached"] == "right"


def test_a_block_in_no_bin_reaches_neither():
    assert where({})["reached"] == "neither"


def test_a_block_claimed_by_both_bins_falls_back_to_the_nearer_centre():
    """Wedged between them or still in the gripper above them -- guessing by overlap
    alone would be a coin toss, so the distance decides and both are recorded."""
    placement = where({"bin_0": True, "bin_1": True}, block=(0.25, 0.50))

    assert placement["reached"] == "right"
    assert placement["distance_to"]["right"] < placement["distance_to"]["left"]


def test_the_measurement_reports_what_it_saw_not_just_a_verdict():
    placement = where({"bin_0": True})

    assert placement["overlaps"] == {"left": True, "right": False}
    assert placement["distance_to"]["left"] == 0.0
    assert placement["position"][:2] == [0.25, -0.4]


class FakeScene:
    """Answers Randomize with one drawn layout."""

    def __init__(self, message):
        self.message = message
        self.randomize = object()
        self.requests = []

    def callService(self, client, request):
        self.requests.append(request)
        return FakeResponse(message=self.message)


def test_the_draw_is_read_from_the_structured_fields_not_the_sentence():
    scene = FakeScene(
        '{"goal": "/bin_1", "target": "/blocks/green_block", '
        '"task": "Put the green block in the right bin."}'
    )

    draw = pg.scene_draw(scene, 0, 7, None)

    assert draw == {
        "colour": "green",
        "side": "right",
        "target": "/blocks/green_block",
        "task": "Put the green block in the right bin.",
    }


def test_the_peek_asks_for_the_seed_it_was_given():
    scene = FakeScene(
        '{"goal": "/bin_0", "target": "/blocks/red_block", '
        '"task": "Put the red block in the left bin."}'
    )

    pg.scene_draw(scene, 0, 42, 3)

    assert scene.requests[0].use_seed is True
    assert scene.requests[0].seed == 42
    assert scene.requests[0].use_zone is True
    assert scene.requests[0].zone == 3


def test_a_reworded_scene_stops_the_probe_instead_of_testing_a_stale_sentence():
    """If the task ever rewords its instruction, the flipped sentence this probe writes
    would be one no policy was trained on -- and the verdict would be meaningless."""
    scene = FakeScene(
        '{"goal": "/bin_0", "target": "/blocks/red_block", '
        '"task": "Place the red cube into the left container."}'
    )

    with pytest.raises(SystemExit, match="wording"):
        pg.scene_draw(scene, 0, 1, None)


CENTRES = {"left": np.array([0.25, -0.40]), "right": np.array([0.25, 0.52])}


def test_an_empty_path_has_no_lean():
    assert pg.lean_of([], CENTRES) is None


def test_lean_is_negative_toward_the_left_bin():
    """lean = closest_left - closest_right, so reaching left makes it negative."""
    reached_left = pg.lean_of([[0.25, -0.40, 1.1]], CENTRES)

    assert reached_left["closest"]["left"] == 0.0
    assert reached_left["lean"] < 0


def test_lean_is_positive_toward_the_right_bin():
    assert pg.lean_of([[0.25, 0.52, 1.1]], CENTRES)["lean"] > 0


def test_lean_takes_the_closest_approach_not_the_last_point():
    """The arm may swing to a bin and come back; the visit is what matters."""
    swung_and_returned = pg.lean_of([[0.25, 0.0, 1.1], [0.25, 0.52, 1.1], [0.25, 0.0, 1.1]],
                                    CENTRES)

    assert swung_and_returned["closest"]["right"] == 0.0


def lean(value):
    return {"closest": {}, "lean": value}


def test_an_arm_that_swings_to_whichever_bin_it_was_told_is_following():
    """Told left it leans left (-0.4); told right it leans right (+0.4)."""
    assert pg.arm_response(lean(-0.4), lean(0.4), 0.05) == ("followed", 0.8)


def test_an_arm_that_moves_identically_is_not_using_the_words():
    assert pg.arm_response(lean(-0.4), lean(-0.4), 0.05)[0] == "same"
    assert pg.arm_response(lean(0.4), lean(0.4), 0.05)[0] == "same"


def test_a_resting_bias_scores_zero_not_fifty_percent():
    """An arm parked nearer the left bin leans left in BOTH trials. Absolute lean
    would call that half-right; the differential calls it what it is."""
    assert pg.arm_response(lean(-0.9), lean(-0.9), 0.05) == ("same", 0.0)


def test_swinging_to_the_wrong_bin_each_time_is_opposed():
    assert pg.arm_response(lean(0.4), lean(-0.4), 0.05)[0] == "opposed"


def test_a_lean_below_the_threshold_is_noise():
    assert pg.arm_response(lean(-0.02), lean(0.02), 0.05)[0] == "same"
    assert pg.arm_response(lean(-0.06), lean(0.06), 0.05)[0] == "followed"


def test_a_missing_path_is_unmeasured_not_a_verdict():
    assert pg.arm_response(None, lean(0.4), 0.05) == ("unmeasured", 0.0)
    assert pg.arm_response(lean(0.4), None, 0.05) == ("unmeasured", 0.0)


class FakeNode:
    def __init__(self):
        self.created = []

    def create_client(self, srv_type, srv_name, callback_group=None):
        self.created.append(srv_name)
        return f"client:{srv_name}"


class FakeClientRobot:
    def __init__(self):
        self.node = FakeNode()
        self._reentrant_callback_group = None


def test_both_scene_clients_are_attached_together():
    """where_is reaches for robot.collision AND robot.pose. debug_rollout once created
    only the second and died at the end of a rollout it had already paid for."""
    robot = FakeClientRobot()

    pg.attach_scene_clients(robot, "/Sim_0")

    assert robot.collision == "client:/Sim_0/CollisionRequest"
    assert robot.pose == "client:/Sim_0/PoseRequest"


def test_attaching_twice_does_not_make_a_second_client():
    robot = FakeClientRobot()

    pg.attach_scene_clients(robot, "/Sim_0")
    pg.attach_scene_clients(robot, "/Sim_0")

    assert len(robot.node.created) == 2
