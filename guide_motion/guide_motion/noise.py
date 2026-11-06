"""Perturbed scripted motion (DART-style demonstrations): a seeded, smooth, mean-zero aim error.

A move's clean path is a straight line, or, after a free junction, a blend that cuts the corner
of the previous move. Control points sit at the quantiles of a density over the path's fraction s,
1 - (1 - min_spacing) x a Gaussian about the middle: dense at a precise end and start, `spacing`
apart mid-move, the same pattern at any speed and length (a free side stays at the mid density). Each one's offset in the path's normal plane is an AR(1) draw biased to
the previous one (rho = exp(-ds / ell)): an error that persists, grows, fades or crosses the line
(an overshoot) by chance, not by a parameter. Its std decays exponentially towards a precise end
or start, sigma (1 - e^(-d / lam)) per side, with lam = decay x v / v_ref so the return onto the
path takes as long at any speed. A Bezier through the points is the offset curve. Pink IK tracks
clean + offset (planner.plan(route=...)) and the OSCBF controller keeps it safe.

A free end keeps its error; the next move inherits it (a free start) in a frame carried along the
path by parallel transport, so position and (u, v) offset stay continuous across moves.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

BEZIER_SAMPLES = 400
MAX_CONTROL_POINTS = 400


@dataclass(frozen=True)
class NoiseParams:
    sigma: float = 0.012  # m: the deviation's largest std
    decay: float = 0.03  # m: lam at v_ref
    v_ref: float = 0.1  # m/s
    ell: float = 0.15  # m: the bias's correlation length (shorter: more overshoots)
    spacing: float = 0.05  # m: the control point spacing mid-move
    min_spacing: float = 0.1  # the spacing at a precise end, as a share of `spacing`
    width: float = 0.35  # the mid-move Gaussian's std, as a share of the move
    p_perturb: float = 0.6  # the chance a move asked to be perturbed is (clean routes stay the mode)
    blend: float = 0.04  # m: the corner a free junction cuts
    speed_scale: tuple = (0.7, 1.3)  # each move's speed factor, drawn uniformly
    aim: float = 0.03  # m: an aimed free end's horizontal miss, uniform over a disc of this radius
    aim_ramp: float = 0.1  # m: the miss builds up over the move's last aim_ramp, fades over the next's first
    aim_late: bool = False  # fade it over the next move's last aim_ramp instead: corrected near that goal
    seed: int | None = None  # the trajectory master seed; None draws one


def transport(a, b, v) -> np.ndarray:
    """v rotated by the minimal rotation taking unit a onto unit b, made normal to b. A reversal
    (a = -b) has no minimal rotation: v then only loses its b component."""
    c, k = float(np.dot(a, b)), np.cross(a, b)
    r = v if c < -0.999 else v * c + np.cross(k, v) + k * np.dot(k, v) / (1.0 + c)
    r = r - b * np.dot(r, b)
    return r / np.linalg.norm(r)


def _bezier(xs, ys, samples=BEZIER_SAMPLES):
    """A Bezier curve through control points (xs[i], ys[i]) sampled at `samples` parameters."""
    n = len(xs)
    i = np.arange(n)
    t = np.linspace(0.0, 1.0, samples)[1:-1, None]
    lc = np.array([math.lgamma(n) - math.lgamma(j + 1) - math.lgamma(n - j) for j in i])
    w = np.exp(lc + i * np.log(t) + (n - 1 - i) * np.log1p(-t))  # Bernstein weights, in log space
    return np.r_[xs[0], w @ xs, xs[-1]], np.vstack([ys[0], w @ ys, ys[-1]])


def _smooth(x) -> float:
    """Smoothstep, 0 at x <= 0 to 1 at x >= 1."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class Route:
    """A clean polyline, a (u, w) frame carried along it, and an offset curve in that frame.

    s in [0, 1] is the clean path's arc-length fraction. ctrl = (s_i, offsets (n, 2), sigma (n,)).
    """

    def __init__(self, points, tangent=None, u=None, ctrl=None):
        points = np.asarray(points, float)
        seg = np.diff(points, axis=0)
        length = np.linalg.norm(seg, axis=1)
        keep = length > 1e-9
        self.points, self.dirs = np.vstack([points[:-1][keep], points[-1]]), seg[keep] / length[keep, None]
        self._cum = np.r_[0.0, np.cumsum(length[keep])]
        self.length = float(self._cum[-1])
        first = self.dirs[0]
        if u is None:
            u = np.cross(first, [0.0, 0.0, 1.0] if abs(first[2]) < 0.9 else [1.0, 0.0, 0.0])
        us = [transport(first if tangent is None else tangent, first, np.asarray(u, float))]
        for a, b in zip(self.dirs[:-1], self.dirs[1:]):
            us.append(transport(a, b, us[-1]))
        self.us = np.array(us)
        self.ctrl, self.s = ctrl, None
        self.aim()
        if ctrl is not None:
            self._bs, self._bn = _bezier(np.asarray(ctrl[0], float), np.asarray(ctrl[1], float))

    def aim(self, start=None, end=None, ramp=0.1, lead=0.0, late=False) -> None:
        """A miss off the clean path, in metres: `start` (carried in) fades out over the first `ramp`
        (at most 80 % of the route; `late`: held, then faded over the `ramp` before the last `lead`, so
        it is corrected close to this route's goal), `end` builds up over the `ramp` before the last
        `lead` (the blend a free end hands over at), so the route ends at goal + end while clean()
        still ends at the goal."""
        self.aim_in = np.zeros(3) if start is None else np.asarray(start, float)
        self.aim_out = np.zeros(3) if end is None else np.asarray(end, float)
        self._lead = min(lead, 0.45 * self.length)
        self._ramp_in, self._ramp_out = min(ramp, 0.8 * self.length), min(ramp, self.length - self._lead)
        self._fade_from = max(self.length - self._lead - self._ramp_in, 0.0) if late else 0.0

    def miss(self, s) -> np.ndarray:
        d = s * self.length
        return (self.aim_in * (1.0 - _smooth((d - self._fade_from) / self._ramp_in))
                + self.aim_out * _smooth(1.0 - (self.length - self._lead - d) / self._ramp_out))

    def _segment(self, s):
        return int(np.clip(np.searchsorted(self._cum, s * self.length, "right") - 1, 0, len(self.dirs) - 1))

    def clean(self, s) -> np.ndarray:
        k = self._segment(s)
        return self.points[k] + self.dirs[k] * (s * self.length - self._cum[k])

    def tangent(self, s) -> np.ndarray:
        return self.dirs[self._segment(s)]

    def frame(self, s):
        k = self._segment(s)
        return self.us[k], np.cross(self.dirs[k], self.us[k])

    def offset(self, s) -> np.ndarray:
        if self.ctrl is None:
            return np.zeros(2)
        return np.array([np.interp(s, self._bs, self._bn[:, 0]), np.interp(s, self._bs, self._bn[:, 1])])

    def sigma(self, s) -> float:
        return 0.0 if self.ctrl is None else float(np.interp(s, self.ctrl[0], self.ctrl[2]))

    def at(self, s) -> np.ndarray:
        (u, w), (a, b) = self.frame(s), self.offset(s)
        return self.clean(s) + a * u + b * w + self.miss(s)


class Noise:
    """One run's motion noise: a master seed, a per-episode key, and the previous move it carries on."""

    def __init__(self, params: NoiseParams):
        self.params = params
        self.master = int(params.seed if params.seed is not None else np.random.SeedSequence().entropy % 2**63)
        self.episode()

    def episode(self, *key, clean=False) -> None:
        """Start an episode: moves are seeded (master, *key, move number), so a key repeats its routes.
        clean: no route is perturbed (speed factors and goal jitter still apply)."""
        self.key, self._move, self.prev, self.clean = tuple(int(k) for k in key), 0, None, clean

    def move(self) -> np.random.Generator:
        """The next move's generator (its speed factor, start offset and route draw from it)."""
        self._move += 1
        return np.random.default_rng(np.random.SeedSequence([self.master, *self.key, self._move]))

    def route(self, rng, p0, goal, v, perturb=True, free_end=False, handover=None, aim=False) -> Route | None:
        """The route from p0 (or, handed over at path parameter `handover` of the previous route, from
        there through a blend) to goal at cruise speed v. None when p0 is the goal. aim (free ends of
        perturbed episodes): the route ends a horizontal miss off the goal, its clean point on it; the
        next move, handed over, fades the miss out."""
        P, prev = self.params, self.prev
        goal = np.asarray(goal, float)
        carry, tangent, u, aim_in = None, None, None, None
        if handover is not None and prev is not None:
            A = prev.route
            start, corner = A.clean(handover), A.clean(1.0)
            out = goal - corner
            r = min(P.blend, 0.45 * np.linalg.norm(out))
            t = np.linspace(0.0, 1.0, 25)[:, None]
            b = corner + r * out / max(np.linalg.norm(out), 1e-9)
            points = np.vstack([(1 - t) ** 2 * start + 2 * t * (1 - t) * corner + t ** 2 * b, goal])
            tangent, u = A.tangent(handover), A.frame(handover)[0]
            if prev.carries:
                carry = (A.offset(handover), A.sigma(handover))
            aim_in = A.miss(handover)
        else:
            points = np.vstack([p0, goal])
            if prev is not None:
                tangent, u = prev.route.tangent(1.0), prev.route.frame(1.0)[0]
        if np.linalg.norm(np.diff(points, axis=0), axis=1).sum() < 1e-6:
            return None
        route = Route(points, tangent, u)
        sigma = P.sigma if perturb and not self.clean and rng.random() < P.p_perturb else 0.0
        if sigma > 0 or carry is not None:
            route = Route(points, tangent, u, self._control(rng, route.length, v, sigma, carry, free_end))
        aim_out = None
        if aim and free_end and not self.clean:
            r, th = P.aim * math.sqrt(rng.random()), rng.uniform(0.0, 2.0 * math.pi)
            aim_out = [r * math.cos(th), r * math.sin(th), 0.0]
        route.aim(aim_in, aim_out, P.aim_ramp, lead=P.blend if free_end else 0.0, late=P.aim_late)
        self.prev = SimpleNamespace(route=route, carries=free_end and route.ctrl is not None)
        return route

    def _spacing(self, L, free_start=False, free_end=False) -> np.ndarray:
        """Control point fractions: the quantiles of 1 / spacing x (1 / min_spacing - (1 / min_spacing - 1)
        x G(s)), G a Gaussian about the middle, held at 1 over a free half (no convergence there)."""
        P = self.params
        x = np.linspace(0.0, 1.0, 1001)
        G = np.exp(-0.5 * ((x - 0.5) / P.width) ** 2)
        G[(x <= 0.5) & free_start] = G[(x >= 0.5) & free_end] = 1.0
        k = 1.0 / P.min_spacing
        cdf = np.r_[0.0, np.cumsum(np.diff(x) * 0.5 * ((k - (k - 1) * G)[1:] + (k - (k - 1) * G)[:-1]))]
        n = int(np.clip(round(cdf[-1] * L / P.spacing), 1, MAX_CONTROL_POINTS - 1))
        return np.interp(np.linspace(0.0, 1.0, n + 1), cdf / cdf[-1], x)

    def _control(self, rng, L, v, sigma, carry, free_end):
        """Control points: the spacing walk, then the AR(1) offsets and their stds."""
        P = self.params
        lam = P.decay * v / P.v_ref
        ramp = lambda x: -math.expm1(-x * L / lam)
        g = lambda s: (1.0 if carry is not None else ramp(s)) * (1.0 if free_end else ramp(1.0 - s))
        ss = self._spacing(L, free_start=carry is not None, free_end=free_end)
        n = len(ss)
        sig = [carry[1] if carry is not None else 0.0] + [sigma * g(x) for x in ss[1:]]
        if not free_end:
            sig[-1] = 0.0
        off = [np.asarray(carry[0], float) if carry is not None else np.zeros(2)]
        for i in range(1, n if free_end else n - 1):
            e = rng.standard_normal(2)
            if sig[i - 1] == 0.0:
                off.append(sig[i] * e)
                continue
            rho = math.exp(-(ss[i] - ss[i - 1]) * L / P.ell)
            off.append(rho * sig[i] / sig[i - 1] * off[-1] + sig[i] * math.sqrt(1.0 - rho * rho) * e)
        if not free_end:
            off.append(np.zeros(2))
        return np.array(ss), np.array(off), np.array(sig)
