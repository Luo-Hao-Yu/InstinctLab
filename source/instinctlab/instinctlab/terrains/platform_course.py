"""Deterministic platform bounds shared by geometry, spawn sampling and scoring."""

import numpy as np


def platform_course_bounds(difficulty, cfg, seed=0):
    """Return N (start, end) pairs in tile-local X; each course has independently sized gaps.

    Widths are quantized to the height-field grid. The hardest allowed gap and a
    minimum-width gap occur in every course, in shuffled positions.
    """
    count, length, scale = cfg.platform_count, cfg.platform_length, cfg.horizontal_scale
    if count < 2 or length <= 0 or cfg.platform_width <= 0:
        raise ValueError("A platform course requires at least two positive-sized platforms.")
    if not 0 <= difficulty <= 1:
        raise ValueError("Course difficulty must be in [0, 1].")
    for size in (length, cfg.platform_width, *cfg.gap_distance_range):
        if not np.isclose(size / scale, round(size / scale)):
            raise ValueError("Platform dimensions and gap limits must be multiples of horizontal_scale.")
    minimum = round(cfg.gap_distance_range[0] / scale)
    maximum = round((cfg.gap_distance_range[0] + difficulty * np.ptp(cfg.gap_distance_range)) / scale)
    if minimum < 3 or maximum < minimum:
        raise ValueError("Course gaps need at least three grid intervals and ordered width limits.")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed or 0), int(round(difficulty * 1e9))]))
    gaps = rng.integers(minimum, maximum + 1, size=count - 1)
    gaps[0] = maximum
    if count > 2:
        gaps[-1] = minimum
    rng.shuffle(gaps)
    # Anchor the first platform; shorter/easier courses finish earlier in the tile.
    max_length = count * length + (count - 1) * cfg.gap_distance_range[1]
    start = -max_length / 2
    starts = start + np.concatenate(([0.0], np.cumsum(length + gaps * scale)))
    return np.stack((starts, starts + length), axis=-1)
