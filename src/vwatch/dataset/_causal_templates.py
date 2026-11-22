"""T3 chain templates. Times are in base units (speed scale 1, chain starts at t=0); the builder
retimes them. Each template returns its actors, background and (cause, effect, time) links."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from vwatch.dataset import _causal_scene as S
from vwatch.dataset import _render as R


@dataclass(frozen=True)
class Built:
    actors: tuple[S.Actor, ...]
    bg: Image.Image
    links: tuple[tuple[str, str, float], ...]
    chain_end: float


def _rgb(name: str) -> S.Rgb:
    return R.COLOURS[name]


def ledge_template(rng: np.random.Generator, cols: list[str], cannon: bool) -> Built:
    """[cannon fires ->] ball rolls along a ledge, hits a box, box falls onto a button, lamp turns on."""
    bc, xc, pc, cc = cols[:4]
    noun = ("box", "block", "crate")[int(rng.integers(3))]
    bg = S.sky_bg()
    d = ImageDraw.Draw(bg)
    d.rectangle((0, 360, 340, 384), fill=(96, 98, 112))
    d.rectangle((16, 384, 44, S.FLOOR_Y), fill=(120, 122, 134))
    x0, v = (95.0, 300.0) if cannon else (60.0, 240.0)
    t_roll = 0.18 if cannon else 0.0
    x_hit = 218.0
    t_hit = t_roll + (x_hit - x0) / v
    t_off = t_hit + 50 / 150
    kl = math.sqrt(2 * (712 - 318) / 1800)
    t_land = t_off + kl
    t_lamp = t_land + 0.35

    def ball_x(t: float) -> float:
        if t < t_roll:
            return x0
        if t < t_hit:
            return x0 + v * (t - t_roll)
        return x_hit + 70 * (1 - math.exp(-4 * (t - t_hit)))

    def box_pos(t: float) -> tuple[float, float]:
        if t < t_hit:
            return 290.0, 318.0
        if t < t_off:
            return 290 + 150 * (t - t_hit), 318.0
        k = t - t_off
        if k < kl:
            return 340 + 55 * k / kl, 318 + 900 * k * k
        return 395.0, 712 + 12 * S.ease_out((t - t_land) / 0.1)

    actors = [
        S.Actor(4, S.const(S.ball_img(_rgb(bc), 30)), lambda t: (ball_x(t), 330.0),
                lambda t: -(ball_x(t) - x0) / 30 * 57.2958),
        S.Actor(5, S.const(S.box_img(_rgb(xc), 84)), box_pos,
                lambda t: -90 * (t - t_off) / kl if t_off <= t < t_land else 0.0),
        S.Actor(3, S.switch([(0, S.button_img(_rgb(pc), False)), (t_land, S.button_img(_rgb(pc), True))]), S.at(395, 770)),
        S.Actor(2, S.switch([(0, S.lamp_img(False)), (t_lamp, S.lamp_img(True))]), S.at(495, 685)),
    ]
    links = [
        (f"the {bc} ball hits the {xc} {noun}", f"the {xc} {noun} falls off the ledge", t_off),
        (f"the {xc} {noun} falls off the ledge", f"the {xc} {noun} lands on the {pc} button", t_land),
        (f"the {xc} {noun} lands on the {pc} button", "the lamp turns on", t_lamp),
    ]
    if cannon:
        muz = S.muzzle_img()
        actors += [S.Actor(6, S.const(S.cannon_img(_rgb(cc))), S.at(55, 333)),
                   S.Actor(7, lambda t: muz if 0 <= t < 0.18 else None, S.at(135, 330))]
        links.insert(0, (f"the {cc} cannon fires", f"the {bc} ball rolls along the ledge", t_roll))
    return Built(tuple(actors), bg, tuple(links), t_lamp + 0.3)


def domino_template(rng: np.random.Generator, cols: list[str], light: bool) -> Built:
    """Ball rolls down a ramp, dominoes topple, lever flips, flag rises [, light turns on]."""
    bc, dc, lc, fc = cols[:4]
    n = int(rng.integers(3, 5))
    bg = S.sky_bg()
    d = ImageDraw.Draw(bg)
    d.polygon([(20, 645), (210, 785), (20, 785)], fill=(110, 112, 124))
    d.polygon([(26, 656), (200, 779), (26, 779)], fill=(150, 152, 162))
    d.rectangle((526, 300, 534, S.FLOOR_Y), fill=(110, 112, 124))
    x_first = 320
    t_hit = 0.8 + (x_first - 12 - 28 - 226) / 300
    starts = [t_hit + 0.14 * i for i in range(n)]
    t_flip = starts[-1] + 0.12
    t_fs = t_flip + 0.35
    t_top = t_fs + 1.4
    t_light = t_top + 0.1
    xl = x_first + 34 * (n - 1) + 62
    contact = x_first - 12 - 28

    def ball_pos(t: float) -> tuple[float, float]:
        if t < 0:
            return 46.0, 629.0
        if t < 0.8:
            u = S.ease_in(t / 0.8)
            return S.lerp(46, 226, u), S.lerp(629, 757, u)
        if t < t_hit:
            return 226 + 300 * (t - 0.8), 757.0
        return contact + 40 * (1 - math.exp(-4 * (t - t_hit))), 757.0

    def domino(i: int) -> S.Actor:
        xi, ti = x_first + 34 * i, starts[i]

        def theta(t: float) -> float:
            return 80 * S.ease_in((t - ti) / 0.45)

        def pos(t: float) -> tuple[float, float]:
            r = math.radians(theta(t))
            return xi + 12 + (-12 * math.cos(r) + 55 * math.sin(r)), 785 + (-12 * math.sin(r) - 55 * math.cos(r))
        return S.Actor(5, S.const(S.rect_img(24, 110, _rgb(dc), 5)), pos, lambda t: -theta(t))

    def lever_theta(t: float) -> float:
        return S.lerp(-30, 30, S.ease_out((t - t_flip) / 0.3))

    def lever_pos(t: float) -> tuple[float, float]:
        r = math.radians(lever_theta(t))
        return xl + 40 * math.sin(r), 785 - 40 * math.cos(r)

    flag = S.poly_img(52, 38, [(52, 0), (0, 19), (52, 38)], _rgb(fc))
    actors = [S.Actor(4, S.const(S.ball_img(_rgb(bc), 28)), ball_pos,
                      lambda t: -(ball_pos(t)[0] - 46) / 28 * 57.2958),
              S.Actor(5, S.const(S.rect_img(16, 80, _rgb(lc), 4)), lever_pos, lambda t: -lever_theta(t)),
              S.Actor(3, S.const(flag), lambda t: (500.0, S.lerp(650, 340, S.ease_out((t - t_fs) / 1.4))))]
    actors += [domino(i) for i in range(n)]
    links = [
        (f"the {bc} ball hits the first {dc} domino", f"the {dc} dominoes fall in a row", t_hit),
        (f"the {dc} dominoes fall in a row", f"the {lc} lever flips", t_flip),
        (f"the {lc} lever flips", f"the {fc} flag goes up the pole", t_fs),
    ]
    end = t_top
    if light:
        actors.append(S.Actor(2, S.switch([(0, S.ball_img((170, 170, 176), 18)), (t_light, S.burst_img((255, 225, 60), 18))]),
                              S.at(530, 280)))
        links.append((f"the {fc} flag goes up the pole", "the light turns on", t_light))
        end = t_light
    return Built(tuple(actors), bg, tuple(links), end + 0.3)


def seesaw_template(rng: np.random.Generator, cols: list[str]) -> Built:
    """A block drops on a seesaw, launching a ball into a balloon which pops."""
    kc, lc, pc = cols[:3]
    bg = S.sky_bg()
    d = ImageDraw.Draw(bg)
    d.polygon([(240, 785), (300, 785), (270, 755)], fill=(110, 112, 124))
    a0, a1 = -14.0, 14.0

    def end(a: float, dist: float) -> tuple[float, float]:
        r = math.radians(a)
        return 270 + dist * math.cos(r), 745 - dist * math.sin(r)

    def up(a: float) -> tuple[float, float]:
        r = math.radians(a)
        return -math.sin(r), -math.cos(r)

    def on_plank(a: float, dist: float, lift: float) -> tuple[float, float]:
        p, n = end(a, dist), up(a)
        return p[0] + n[0] * lift, p[1] + n[1] * lift

    bx0, by0 = on_plank(a0, -125, 50)
    t_land = math.sqrt((by0 + 60) / 900)
    t_launch = t_land + 0.2

    def angle(t: float) -> float:
        return a0 if t < t_land else S.lerp(a0, a1, S.ease_out((t - t_land) / 0.2))

    def block_pos(t: float) -> tuple[float, float]:
        if t < t_land:
            return bx0, -60 + 900 * max(t, 0.0) ** 2
        return on_plank(angle(t), -125, 50)

    x0, y0 = on_plank(a1, 130, 36)
    bx, by = float(rng.uniform(430, 490)), float(rng.uniform(250, 380))
    tf = 0.9
    vx, vy = (bx - x0) / tf, (by - y0 - 900 * tf**2) / tf

    def flight(t: float) -> tuple[float, float]:
        k = t - t_launch
        return x0 + vx * k, y0 + vy * k + 900 * k * k

    def ball_pos(t: float) -> tuple[float, float]:
        return on_plank(angle(t), 130, 36) if t < t_launch else flight(t)

    t_pop = next((t_launch + i * 0.005 for i in range(int(tf / 0.005) + 1)
                  if math.hypot(flight(t_launch + i * 0.005)[0] - bx, flight(t_launch + i * 0.005)[1] - by) <= 60), t_launch + tf)
    balloon, burst = S.balloon_img(_rgb(pc), 38), S.burst_img(_rgb(pc), 34)
    actors = [
        S.Actor(3, S.const(S.rect_img(300, 20, (88, 92, 110), 5)), S.at(270, 745), angle),
        S.Actor(5, S.const(S.box_img(_rgb(kc), 80)), block_pos),
        S.Actor(4, S.const(S.ball_img(_rgb(lc), 26)), ball_pos),
        S.Actor(6, S.switch([(-1e9, balloon), (t_pop, burst), (t_pop + 0.25, None)]),
                lambda t: (bx, by) if t >= t_pop else (bx, by + 20 + 4 * math.sin(2 * math.pi * t / 2.5))),
    ]
    links = (
        (f"the {kc} block lands on the seesaw", f"the {lc} ball is launched into the air", t_launch),
        (f"the {lc} ball hits the {pc} balloon", f"the {pc} balloon pops", t_pop),
    )
    return Built(tuple(actors), bg, links, t_pop + 0.4)


def distractor(rng: np.random.Generator, kind: str, colour: str, total: float) -> tuple[str, S.Actor]:
    """A sky-band event that causes nothing: returns (phrase, actor in global time)."""
    dur = float(rng.uniform(4.0, 7.0))
    start = float(rng.uniform(0.5, max(0.6, total - dur - 0.5)))
    y0 = float(rng.uniform(70, 170))
    xa, xb = (-90.0, 630.0) if rng.random() < 0.5 else (630.0, -90.0)
    if kind == "bird":
        wings = (S.bird_img(_rgb(colour), False), S.bird_img(_rgb(colour), True))
        phrase = f"a {colour} bird flies across the top"
        img = lambda t: wings[int((t - start) / 0.12) % 2] if start <= t < start + dur else None  # noqa: E731
    elif kind == "kite":
        kite = S.kite_img(_rgb(colour))
        phrase = f"a {colour} kite floats across the top"
        img = lambda t: kite if start <= t < start + dur else None  # noqa: E731
    else:
        cloud = S.cloud_img()
        phrase = "a white cloud drifts across the top"
        img = lambda t: cloud if start <= t < start + dur else None  # noqa: E731
    pos = lambda t: (S.lerp(xa, xb, (t - start) / dur), y0 + 10 * math.sin(2 * math.pi * (t - start) / 1.6))  # noqa: E731
    return phrase, S.Actor(1, img, pos)
