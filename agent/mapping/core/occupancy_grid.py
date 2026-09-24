import os

import numpy as np


def disc_offsets(r_cells):
    """(K, 2) int array of (dx, dz) offsets with dx^2 + dz^2 <= r_cells^2, row-major order."""
    d = np.arange(-r_cells, r_cells + 1)
    dx, dz = np.meshgrid(d, d, indexing="ij")
    inside = dx * dx + dz * dz <= r_cells * r_cells
    return np.stack([dx[inside], dz[inside]], axis=1)


def add_disc(array, cx, cz, r_cells, value, *extra_index):
    """array[cx+dx, cz+dz, *extra_index] += value over the in-bounds disc of radius r_cells."""
    offs = disc_offsets(r_cells)
    nx, nz = cx + offs[:, 0], cz + offs[:, 1]
    ok = (nx >= 0) & (nx < array.shape[0]) & (nz >= 0) & (nz < array.shape[1])
    array[(nx[ok], nz[ok]) + extra_index] += value


class OccupancyGrid:
    """2D log-odds occupancy grid over the world XZ (top-down) plane."""

    FREE_THRESHOLD = -0.5
    OCCUPIED_THRESHOLD = 0.5
    FREE_DECREMENT = 0.4
    OCCUPIED_INCREMENT = 0.85

    def __init__(self, size_m=60.0, resolution=0.1):
        self.res = resolution
        self.n = int(size_m / resolution)
        self.origin = -size_m / 2.0
        self.log_odds = np.zeros((self.n, self.n), dtype=np.float32)

    @classmethod
    def from_log_odds(cls, log_odds, resolution):
        """Wrap a saved log-odds array (square, centred on the origin) in a grid."""
        grid = cls(size_m=log_odds.shape[0] * resolution, resolution=resolution)
        grid.log_odds = log_odds
        return grid

    def cell(self, wx, wz):
        return int((wx - self.origin) / self.res), int((wz - self.origin) / self.res)

    def to_world(self, cx, cz):
        return self.origin + (cx + 0.5) * self.res, self.origin + (cz + 0.5) * self.res

    def in_bounds(self, cx, cz):
        return 0 <= cx < self.n and 0 <= cz < self.n

    def integrate(self, sensor_world_xz, hit_points_xz):
        sx, sz = self.cell(*sensor_world_xz)
        for wx, wz in hit_points_xz:
            hx, hz = self.cell(wx, wz)
            if not self.in_bounds(hx, hz):
                continue
            for cx, cz in _bresenham_line(sx, sz, hx, hz)[:-1]:
                if self.in_bounds(cx, cz):
                    self.log_odds[cx, cz] -= self.FREE_DECREMENT
            self.log_odds[hx, hz] += self.OCCUPIED_INCREMENT

    def mark_occupied(self, world_xz):
        cx, cz = self.cell(*world_xz)
        if self.in_bounds(cx, cz):
            self.log_odds[cx, cz] += self.OCCUPIED_INCREMENT

    def mark_occupied_region(self, world_xz, radius_m):
        """Mark every cell within radius_m of world_xz occupied, so one block registers the
        obstacle's whole footprint for inflated A* instead of a single cell."""
        cx, cz = self.cell(*world_xz)
        add_disc(self.log_odds, cx, cz, int(round(radius_m / self.res)), self.OCCUPIED_INCREMENT)

    def frontier_mask(self):
        """Free cells 4-adjacent to an unknown cell. Unknown is ~free & ~occupied, so weakly
        observed cells stay frontier candidates; False padding avoids np.roll wraparound."""
        free = self.log_odds < self.FREE_THRESHOLD
        occupied = self.log_odds > self.OCCUPIED_THRESHOLD
        unknown = ~free & ~occupied
        padded = np.pad(unknown, 1, mode="constant", constant_values=False)
        touches_unknown = np.zeros_like(unknown)
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            touches_unknown |= padded[1 + dx : 1 + dx + self.n, 1 + dz : 1 + dz + self.n]
        return free & touches_unknown

    def frontiers(self):
        return np.argwhere(self.frontier_mask())


def load_grid(output_dir, tag, resolution):
    """Load grid_<tag>.npy from output_dir as an OccupancyGrid."""
    return OccupancyGrid.from_log_odds(np.load(os.path.join(output_dir, f"grid_{tag}.npy")), resolution)


def draw_grid(ax, grid, margin_m=1.0):
    """imshow the grid (white free, black occupied, grey unknown), cropped to the known
    region plus margin_m. Axes are cell indices (X horizontal, Z vertical)."""
    display = np.full(grid.log_odds.shape, 0.5)
    display[grid.log_odds < grid.FREE_THRESHOLD] = 1.0
    display[grid.log_odds > grid.OCCUPIED_THRESHOLD] = 0.0
    ax.imshow(display.T, origin="lower", cmap="gray", vmin=0, vmax=1)
    known_cells = np.argwhere(display.T != 0.5)
    if len(known_cells) > 0:
        margin = int(round(margin_m / grid.res))
        y0, x0 = known_cells.min(axis=0) - margin
        y1, x1 = known_cells.max(axis=0) + margin
        ax.set_xlim(max(0, x0), min(grid.log_odds.shape[0], x1))
        ax.set_ylim(max(0, y0), min(grid.log_odds.shape[1], y1))


def _bresenham_line(x0, z0, x1, z1):
    points = []
    dx, dz = abs(x1 - x0), abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sz = 1 if z0 < z1 else -1
    err = dx - dz
    x, z = x0, z0
    while True:
        points.append((x, z))
        if x == x1 and z == z1:
            break
        e2 = 2 * err
        if e2 > -dz:
            err -= dz
            x += sx
        if e2 < dx:
            err += dx
            z += sz
    return points
