"""Regression tests for audited mapping bugs (topology dedup, vantage arrival, reconciler,
LiDAR loop threading, annotate_pass helpers). No sim needed."""
import asyncio
import importlib
import os
import struct
import sys
import threading
from types import SimpleNamespace

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAPPING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))), "agent", "mapping")
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402

import annotate_pass  # noqa: E402
import lidar_client  # noqa: E402
import topology as topo  # noqa: E402
from frontier_planner import FrontierCluster, FrontierPlanner  # noqa: E402
from occupancy_grid import OccupancyGrid  # noqa: E402


def test_trace_edges_keeps_edges_whose_ids_and_lengths_collide():
    # (2,5,len 3) and (3,5,len 2) both collapsed to frozenset {2,3,5} under the old key.
    node_of = {(10, 8): 2, (10, 10): 5, (10, 11): 3}
    skeleton = set(node_of) | {(10, 9)}
    edges = topo._trace_edges(skeleton, node_of, connectivity=8)
    assert {frozenset((a, b)) for a, b, _ in edges} == {frozenset((2, 5)), frozenset((3, 5))}


def test_vantage_commit_judges_arrival_at_the_vantage():
    grid = OccupancyGrid(size_m=10.0, resolution=0.1)
    planner = FrontierPlanner(grid)
    cluster = FrontierCluster(cells=np.array([[50, 50]]), centroid_cell=(50, 50), size=4)
    vantage = (70, 50)
    planner._plan_to_cluster = lambda cur, c: ([grid.to_world(60, 50), grid.to_world(*vantage)], vantage)

    nav = planner._plan_first_reachable((60, 50), [cluster])

    assert planner.goal_cell == vantage
    assert nav.goal_world_xz == grid.to_world(*vantage)
    # standing at the vantage counts as arrival -> replan (not an 8-step no-progress stall)
    planner._pick_and_plan = lambda cur: "replanned"
    assert planner.update(grid.to_world(*vantage), vantage) == "replanned"


def test_reconciler_does_not_match_an_empty_name_to_every_sku(monkeypatch, tmp_path):
    monkeypatch.setenv("SARI_SANDBOX_DIR", str(tmp_path))
    rp = importlib.import_module("reconcile_products")
    rec = rp.Reconciler.__new__(rp.Reconciler)
    rec.sku_raw = {"RC_COLA": "rccola", "PIATTOS": "piattos"}
    result = rec.match("!!!")
    assert result["sku"] is None and result["sku_candidates"] is None


def _lidar_payload():
    header = struct.pack(lidar_client.HEADER_FORMAT, lidar_client.MAGIC, 1, 1,
                         0.05, 20.0, 0.0, 1.0, 7, 0.0)
    return header + struct.pack("<f", 0.0) + struct.pack("<f", 3.0)


def test_request_lidar_scan_works_off_the_main_thread(monkeypatch):
    async def fake_request(_uri):
        return _lidar_payload()

    monkeypatch.setattr(lidar_client, "_request_scan_async", fake_request)
    out = {}

    def worker():
        try:
            out["scan"] = lidar_client.RequestLidarScan("ws://x", timeout=5.0)
        except Exception as error:  # surfaced below
            out["error"] = error

    t = threading.Thread(target=worker)
    t.start()
    t.join(10)
    assert "error" not in out, out.get("error")
    assert out["scan"]["ranges"] == [3.0]


def test_thread_loop_is_reused_within_a_thread():
    loop = lidar_client._thread_loop()
    assert isinstance(loop, asyncio.AbstractEventLoop)
    assert lidar_client._thread_loop() is loop


def test_select_checkpoints_does_not_reorder_the_topology():
    topology = {"checkpoints": [{"id": 3, "kind": "shelf"}, {"id": 1, "kind": "end"}]}
    args = SimpleNamespace(kind="all", ids=None, limit=0)
    picked = annotate_pass.select_checkpoints(topology, args)
    assert [c["id"] for c in picked] == [1, 3]
    assert [c["id"] for c in topology["checkpoints"]] == [3, 1]


def test_render_semantic_map_tolerates_null_summary():
    annotations = {"1": {"effective_kind": "non_shelf", "annotation": {"semantic_summary": None}}}
    text = annotate_pass.render_semantic_map(annotations, {"checkpoints": [{"id": 1}]})
    assert "Checkpoint 1" in text
