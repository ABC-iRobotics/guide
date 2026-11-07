import numpy as np

from guide_motion.noise import Noise, NoiseParams
from guide_motion.planner import DT, profile

P0, GOAL, NEXT = np.array([0.3, 0.0, 0.4]), np.array([0.47, 0.13, 0.3]), np.array([0.47, 0.13, 0.04])


def draw(seed=1, key=(0, 0), **kw):
    noise = Noise(NoiseParams(seed=seed, **{"p_perturb": 1.0, **kw}))
    noise.episode(*key)
    return noise, noise.route(noise.move(), P0, GOAL, v=0.25)


def test_precise_ends_and_normal_offsets():
    _, r = draw()
    assert np.allclose(r.at(0.0), P0) and np.allclose(r.at(1.0), GOAL)
    for s in np.linspace(0, 1, 21):
        assert abs(np.dot(r.at(s) - r.clean(s), r.tangent(s))) < 1e-9  # in the normal plane
    assert max(np.linalg.norm(r.offset(s)) for s in np.linspace(0, 1, 21)) > 1e-4


def test_off_and_seeding():
    _, r = draw(sigma=0.0)
    assert all(np.allclose(r.at(s), r.clean(s)) for s in np.linspace(0, 1, 11))
    a, b, c = draw(key=(3, 7))[1], draw(key=(3, 7))[1], draw(key=(3, 8))[1]
    assert np.allclose(a.at(0.5), b.at(0.5)) and not np.allclose(a.at(0.5), c.at(0.5))


def test_std_vanishes_at_precise_ends():
    dev = np.array([[np.linalg.norm(draw(seed=k)[1].offset(s)) for s in (0.02, 0.5, 0.98)] for k in range(200)])
    rms = np.sqrt((dev ** 2).mean(axis=0))
    assert rms[1] > 0.006 and rms[0] < 0.3 * rms[1] and rms[2] < 0.3 * rms[1]


def test_free_junction_is_continuous():
    noise = Noise(NoiseParams(seed=5, p_perturb=1.0))
    a = noise.route(noise.move(), P0, GOAL, v=0.25, free_end=True)
    assert np.linalg.norm(a.offset(1.0)) > 0  # the free end keeps its error
    h = 1.0 - 0.03 / a.length  # handed over 3 cm before A's goal
    b = noise.route(noise.move(), None, NEXT, v=0.1, handover=h)
    # the blend polyline's first chord turns the frame slightly: sub-millimetre
    assert np.allclose(b.at(0.0), a.at(h), atol=1e-3) and np.allclose(b.offset(0.0), a.offset(h))
    assert np.allclose(b.frame(0.0)[0], a.frame(h)[0], atol=0.05)
    assert np.allclose(b.at(1.0), NEXT)  # a precise end after a free start
    assert np.linalg.norm(b.clean(0.5) - (GOAL + 0.5 * (NEXT - GOAL))) > 0  # the corner is cut


def test_profile_free_start_and_end():
    s = profile(0.3, 0.25, stop=False)
    assert (s[-2] - s[-3]) / DT > 0.24  # no deceleration onto a free end (the last step is the remainder)
    s = profile(0.3, 0.25, v0=0.2)
    assert s[0] / DT > 0.18  # starts at the handed-over speed


def test_plan_tracks_the_route_with_pink(fr3_model):
    from fr3_fixture import HOME
    from guide_motion.planner import plan
    from guide_motion.targets import Target

    p, R = fr3_model.ee_pose(HOME)
    noise = Noise(NoiseParams(seed=2, sigma=0.015, p_perturb=1.0))
    route = noise.route(noise.move(), p, p + [0.15, 0.1, -0.15], v=0.25)
    targets = plan(fr3_model, Target(position=p + [0.15, 0.1, -0.15], rotation=R), HOME, speed=0.5, route=route)
    assert len(targets) == len(route.s)
    assert max(np.linalg.norm(t.position - route.at(s)) for t, s in zip(targets, route.s)) < 2e-3
    assert all(np.allclose(t.clean, route.clean(s)) for t, s in zip(targets, route.s))
    assert np.allclose(targets[-1].position, route.at(1.0))


def test_control_points_crowd_at_converging_ends_at_any_speed():
    noise = Noise(NoiseParams(seed=1))
    for L in (0.1, 0.4):
        ds = np.diff(noise._spacing(L)) * L
        assert ds[0] < 0.3 * ds.max() and ds[-1] < 0.3 * ds.max()  # dense at both precise ends
        assert ds.max() <= 0.05 + 1e-9 and ds[len(ds) // 2] > 0.6 * ds.max()  # the middle keeps points
    ds = np.diff(noise._spacing(0.4, free_end=True)) * 0.4
    assert ds[0] < 0.3 * ds.max() and ds[-1] > 0.8 * ds.max()  # a free end does not converge


def test_p_perturb_leaves_some_moves_clean():
    noise = Noise(NoiseParams(seed=3))
    clean = sum(noise.route(noise.move(), P0, GOAL, v=0.25).ctrl is None for _ in range(200))
    assert 50 < clean < 110  # ~40 % of 200


def test_an_aimed_free_end_misses_its_clean_point_and_the_next_move_corrects():
    noise = Noise(NoiseParams(seed=9, sigma=0.0))  # the miss alone
    a = noise.route(noise.move(), P0, GOAL, v=0.25, free_end=True, aim=True)
    h = 1.0 - noise.params.blend / a.length  # handed over at the blend, as ROS2Robot does
    miss = a.at(h) - a.clean(h)
    assert 0 < np.linalg.norm(miss) <= noise.params.aim and miss[2] == 0  # horizontal, full at the hand-over
    assert np.allclose(a.at(0.0), P0)
    b = noise.route(noise.move(), None, NEXT, v=0.1, handover=h)
    assert np.allclose(b.at(0.0), a.at(h), atol=1e-3)  # continuous
    assert np.allclose(b.at(1.0), NEXT) and np.allclose(b.at(0.8), b.clean(0.8))  # faded before the end
    noise.episode(clean=True)
    assert np.allclose(noise.route(noise.move(), P0, GOAL, v=0.25, free_end=True, aim=True).miss(1.0), 0)


def test_a_late_aim_is_held_down_the_next_move_and_corrected_near_its_goal():
    misses = {}
    for late in (False, True):
        noise = Noise(NoiseParams(seed=9, sigma=0.0, aim_late=late))
        a = noise.route(noise.move(), P0, GOAL, v=0.25, free_end=True, aim=True)
        h = 1.0 - noise.params.blend / a.length
        b = noise.route(noise.move(), None, NEXT, v=0.1, handover=h)  # 26 cm down: fade over the last 10
        misses[late] = [np.linalg.norm(b.miss(s)) for s in (0.0, 0.5, 1.0)]
    full = misses[True][0]
    assert full > 0 and misses[False][1] < 1e-9  # early: gone by mid-descent
    assert np.isclose(misses[True][1], full) and misses[True][2] < 1e-9  # late: held, corrected at the end


def test_a_late_aim_on_a_free_descent_carries_its_rest_into_the_final_descent():
    """Hover -> pre-grasp (free end, blends) -> grasp: the miss is not gone at the blend, it fades to 0 at the grasp."""
    noise = Noise(NoiseParams(seed=9, sigma=0.0, aim_late=True))
    hover, pre, grasp = np.array([0.4, 0.0, 0.2]), np.array([0.4, 0.0, 0.05]), np.array([0.4, 0.0, 0.0])
    a = noise.route(noise.move(), P0, hover, v=0.25, free_end=True, aim=True)
    b = noise.route(noise.move(), None, pre, v=0.1, free_end=True, handover=1.0 - noise.params.blend / a.length)
    h = 1.0 - noise.params.blend / b.length
    c = noise.route(noise.move(), None, grasp, v=0.05, handover=h)
    full = np.linalg.norm(b.miss(0.0))
    assert full > 0 and 0.2 * full < np.linalg.norm(b.miss(h)) < full  # rest left at the blend, 9 cm up
    assert np.allclose(c.aim_in, b.miss(h)) and np.linalg.norm(c.miss(1.0)) < 1e-9  # clean at the grasp
