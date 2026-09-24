import math

import numpy as np

SENSOR_HEIGHT_OFFSET_M = 1.485
"""Meters the LiDAR sensor sits above the agent's reported root position
(world_pos[1], i.e. what TransformAgent reports as translation). Unity's
AlignLevelToSource (LidarSensor.cs) re-parents the sensor to the live
camera/head position each scan, not the root transform that Python reads
back - empirically, the rig's Head Pivot sits ~1.485m above root (local Y
1.632 under a 0.91-scaled "IK Humanoid Agent" prefab root). Rig-specific;
re-derive/verify against a live scan next to a known-height object if the
character model or scale ever changes.
"""

SELF_EXCLUSION_RANGE_M = 0.1
"""Meters within which LiDAR hits are ignored regardless of scan["min_range"] -
distinct from the sensor's own hardware min-range spec, this is a minimal
sensor-housing-noise floor only.

DELIBERATELY kept small. An earlier version of this constant was 0.8m,
intended to blanket-exclude a suspected self-hit (a stowed hand/arm/shoulder
observed at ~0.6-0.7m, sensor-mount height, lateral offset landing right at
body_radius). That was confirmed live to be actively unsafe: with the
blind spot that wide, the agent walked into and scraped along a real shelf
without swept_clearance_ahead ever reporting reduced clearance - the shelf
fell inside the same 0-0.8m dead zone as the suspected self-hit. A blanket
range-based exclusion cannot distinguish "the agent's own body" from "a real
obstacle at a similar distance," so it was rolled back. Do not raise this
back toward 0.6-0.8m without a more targeted (non-range-only) self-hit
signature - e.g. matching the specific v_deg/lateral combination observed -
or a live-confirmed source (try --no-stow-hands and compare the debug output)
so real close obstacles stay detectable.
"""


def normalize_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def angle_to_deg(dx, dz):
    return math.degrees(math.atan2(dx, dz))


def _hit_height_above_root(v_deg, r, sensor_height_offset=SENSOR_HEIGHT_OFFSET_M):
    """Height of a LiDAR hit relative to the agent's own root position (NOT
    absolute world-Y-from-floor - self-relative height sidesteps needing to
    know where the true floor sits relative to the reported root)."""
    return r * math.sin(math.radians(v_deg)) + sensor_height_offset


def _scan_arrays(scan):
    """(ranges[ch, az], v_deg[ch, 1], az_deg[az]) as float arrays, in scan order."""
    channels, samples = scan["channels"], scan["azimuth_samples"]
    ranges = np.asarray(scan["ranges"][:channels * samples], dtype=float).reshape(channels, samples)
    v_deg = np.asarray(scan["vertical_angles_deg"][:channels], dtype=float).reshape(channels, 1)
    az_deg = scan["azimuth_start_deg"] + np.arange(samples) * scan["azimuth_step_deg"]
    return ranges, v_deg, az_deg


def _to_world_xz(lx, lz, world_pos, yaw_deg):
    """Rotate sensor-local (x, z) by Unity's left-handed Euler(0, yaw, 0) and offset."""
    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return lx * cos_y + lz * sin_y + world_pos[0], -lx * sin_y + lz * cos_y + world_pos[2]


def iter_banded_hits(scan, world_pos, yaw_deg, min_obstacle_height=0.05,
                     max_obstacle_height=2.0, sensor_height_offset=SENSOR_HEIGHT_OFFSET_M,
                     self_exclusion_range=SELF_EXCLUSION_RANGE_M):
    """Yield (wx, wz, height_above_root) for every hit that clears the range/self-exclusion
    filters and falls within [min_obstacle_height, max_obstacle_height] of the agent's root.

    Shared by scan_to_world_points() and VoxelGrid.integrate so the rotation convention and
    height band live in one place. Horizontal placement uses the ground-projected distance
    r*cos(v), not the slant range r (slant range smeared steep channels' hits too far out).
    """
    ranges, v_deg, az_deg = _scan_arrays(scan)
    v = np.radians(v_deg)
    exclusion_range = max(scan.get("min_range", 0.0), self_exclusion_range)
    height = ranges * np.sin(v) + sensor_height_offset
    keep = ((ranges < scan["max_range"] - 1e-3) & (ranges > exclusion_range + 1e-3)
            & (height >= min_obstacle_height) & (height <= max_obstacle_height))
    horiz = (ranges * np.cos(v))[keep]  # ground-projected distance, not slant range
    az = np.radians(np.broadcast_to(az_deg, ranges.shape)[keep])
    wx, wz = _to_world_xz(np.sin(az) * horiz, np.cos(az) * horiz, world_pos, yaw_deg)
    yield from zip(wx.tolist(), wz.tolist(), height[keep].tolist())


def scan_to_world_points(scan, world_pos, yaw_deg, min_obstacle_height=0.05,
                          max_obstacle_height=2.0, sensor_height_offset=SENSOR_HEIGHT_OFFSET_M,
                          self_exclusion_range=SELF_EXCLUSION_RANGE_M):
    """Project LiDAR hits whose height falls within [min_obstacle_height,
    max_obstacle_height] of the agent's own body (see _hit_height_above_root)
    into world XZ, for the 2D occupancy grid. Height-band filtering (rather
    than a fixed vertical angle) means overhanging ceiling signage - out of
    the agent's real reach - doesn't get baked into the map as an obstacle,
    while a small min_obstacle_height keeps ordinary floor hits from
    downward-angled channels from doing the same at the low end.

    Thin wrapper over iter_banded_hits() (which owns the projection/filter
    logic, shared with voxel_grid); this one just drops the height component.
    """
    return [
        (wx, wz)
        for wx, wz, _height in iter_banded_hits(
            scan, world_pos, yaw_deg,
            min_obstacle_height=min_obstacle_height, max_obstacle_height=max_obstacle_height,
            sensor_height_offset=sensor_height_offset, self_exclusion_range=self_exclusion_range,
        )
    ]


def scan_to_world_points_3d(scan, world_pos, yaw_deg, vertical_band_deg=90.0,
                             sensor_height_offset=SENSOR_HEIGHT_OFFSET_M):
    """Project LiDAR hits into world XYZ, keeping each channel's own height (point-cloud
    visualization). Unlike scan_to_world_points() there is no height band, only
    vertical_band_deg; same rotation convention."""
    ranges, v_deg, az_deg = _scan_arrays(scan)
    v = np.radians(v_deg)
    keep = (ranges < scan["max_range"] - 1e-3) & (np.abs(v_deg) <= vertical_band_deg)
    horiz = (ranges * np.cos(v))[keep]
    wy = (ranges * np.sin(v) + sensor_height_offset)[keep] + world_pos[1]
    az = np.radians(np.broadcast_to(az_deg, ranges.shape)[keep])
    wx, wz = _to_world_xz(np.sin(az) * horiz, np.cos(az) * horiz, world_pos, yaw_deg)
    return list(zip(wx.tolist(), wy.tolist(), wz.tolist()))


def clearance_ahead(scan, heading_deg, cone_deg=20.0, vertical_band_deg=40.0):
    """Minimum LiDAR range within a cone around heading_deg, where heading_deg
    is relative to the sensor's own forward axis (0 = straight ahead) — pass
    the about-to-be-applied yaw delta, since ranges/azimuth in the scan are
    already sensor-local. Uses a wider vertical band than the 2D map
    projection so shelf items above/below the mapping slice still count.

    Kept for callers that want a plain angular check; prefer
    swept_clearance_ahead() for step-size gating, since a fixed-degree cone
    corresponds to a shrinking linear margin as range decreases and can miss
    obstacles that would clip the agent's actual body width up close.
    """
    channels = scan["channels"]
    azimuth_samples = scan["azimuth_samples"]
    ranges = scan["ranges"]
    min_seen = scan["max_range"]

    for ch in range(channels):
        v_deg = scan["vertical_angles_deg"][ch]
        if abs(v_deg) > vertical_band_deg:
            continue
        row_start = ch * azimuth_samples
        for az_i in range(azimuth_samples):
            az = scan["azimuth_start_deg"] + az_i * scan["azimuth_step_deg"]
            if abs(normalize_deg(az - heading_deg)) > cone_deg:
                continue
            r = ranges[row_start + az_i]
            if r < min_seen:
                min_seen = r
    return min_seen


def swept_clearance_ahead(
    scan, heading_deg, body_radius=0.3, min_obstacle_height=0.05, max_obstacle_height=2.0,
    sensor_height_offset=SENSOR_HEIGHT_OFFSET_M, max_lateral_deg=70.0,
    self_exclusion_range=SELF_EXCLUSION_RANGE_M, debug=False
):
    """Minimum forward distance to any in-band LiDAR hit within body_radius of the straight
    line along heading_deg - clearance for a swept cylinder, not an angular cone (a cone's
    linear coverage shrinks with range and misses close off-axis obstacles).

    Hits within max(scan["min_range"], self_exclusion_range) are excluded (see
    SELF_EXCLUSION_RANGE_M). If debug=True, returns (min_seen, debug_info) describing the
    closest counted hit, or None if nothing counted.
    """
    ranges, v_deg, az_deg = _scan_arrays(scan)
    max_range = scan["max_range"]
    exclusion_range = max(scan.get("min_range", 0.0), self_exclusion_range)
    rel_deg = np.broadcast_to(normalize_deg(az_deg - heading_deg), ranges.shape)
    height = ranges * np.sin(np.radians(v_deg)) + sensor_height_offset
    rel = np.radians(rel_deg)
    forward = ranges * np.cos(rel)
    lateral = ranges * np.sin(rel)
    counted = ((np.abs(rel_deg) <= max_lateral_deg)
               & (ranges < max_range - 1e-3) & (ranges > exclusion_range + 1e-3)
               & (height >= min_obstacle_height) & (height <= max_obstacle_height)
               & (forward > 0) & (np.abs(lateral) <= body_radius)
               & (forward < max_range))
    if not counted.any():
        return (max_range, None) if debug else max_range
    idx = np.unravel_index(np.argmin(np.where(counted, forward, np.inf)), ranges.shape)
    min_seen = float(forward[idx])
    if debug:
        ch = int(idx[0])
        return min_seen, {
            "channel": ch, "v_deg": float(v_deg[ch, 0]), "range": float(ranges[idx]),
            "height_above_root": float(height[idx]), "az_rel_deg": float(rel_deg[idx]),
            "lateral": float(lateral[idx]),
        }
    return min_seen
