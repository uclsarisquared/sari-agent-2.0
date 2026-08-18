"""Regression coverage for an authoritative spawn jump during a frozen-map route."""

import math
from types import SimpleNamespace

from nav import store_map


capture_walk = store_map.capture_walk


class _LineGrid:
    """Minimal integer-cell grid for a straight world-X route."""

    @staticmethod
    def cell(x, _z):
        return (round(x), 0)

    @staticmethod
    def to_world(x, z):
        return (float(x), float(z))


def test_goto_replans_from_spawn_after_mid_route_recovery(monkeypatch):
    planned_starts = []

    def astar(_grid, start, goal, **_kwargs):
        planned_starts.append(start)
        step = 1 if goal[0] >= start[0] else -1
        return [(x, 0) for x in range(start[0], goal[0] + step, step)]

    position = [1.0, 0.0, 0.0]
    yaw = 90.0
    translated = 0

    def step_agent(delta_translation, delta_rotation, _uri):
        nonlocal yaw, translated
        yaw += delta_rotation[1]
        if delta_translation[2]:
            translated += 1
            if translated == 1:
                # The move crossed the sandbox boundary. Its reply is already the authoritative
                # spawn pose, exactly as the recovery protocol guarantees.
                position[0], position[2] = 0.0, 0.0
            else:
                radians = math.radians(yaw)
                position[0] += math.sin(radians) * delta_translation[2]
                position[2] += math.cos(radians) * delta_translation[2]
        return tuple(position), (0.0, yaw, 0.0), False

    monkeypatch.setattr(capture_walk, "astar", astar)
    monkeypatch.setattr(capture_walk, "simplify_path", lambda path, *_a, **_k: path)
    monkeypatch.setattr(capture_walk, "RequestLidarScan", lambda _uri: {})
    monkeypatch.setattr(
        capture_walk, "swept_clearance_ahead", lambda *_a, **_k: (10.0, None)
    )
    monkeypatch.setattr(capture_walk, "step_agent", step_agent)

    args = SimpleNamespace(
        max_steps_per_leg=6,
        arrival_radius=0.01,
        waypoint_arrival_radius=0.01,
        connectivity=4,
        uri="unused",
        body_radius=0.3,
        min_obstacle_height=0.05,
        max_obstacle_height=2.0,
        sensor_height_offset=1.485,
        self_exclusion_range=0.1,
        step_size=1.0,
        safety_margin=0.1,
        min_step=0.05,
        max_nudge_deg=30.0,
        nudge_step_deg=5.0,
        escape_sweep_step_deg=10.0,
    )

    pos, _rot, arrived = capture_walk.goto(
        args, _LineGrid(), object(), (2.0, 0.0), tuple(position), (0.0, yaw, 0.0)
    )

    assert arrived is True
    assert pos[0] == 2.0
    assert planned_starts[:3] == [(1, 0), (0, 0), (1, 0)]
