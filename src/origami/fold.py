"""FOLD format export/import for OrigamiLib."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Callable, TYPE_CHECKING

import networkx as nx
import numpy as np

from .crease_pattern import CreasePattern
from .DCEL import Face, HalfEdge, Plane, Vertex

if TYPE_CHECKING:
    from .core import OrigamiCore


_FOLD_ANGLE: dict[str, int] = {
    "V": 180,
    "M": -180,
    "B": 0,
    "F": 0,
    "U": 0,
}


def _norm_coord(v: float) -> int | float:
    f = float(v)
    rounded = round(f, 10)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _get_face_orders(origami: OrigamiCore, sorted_fids: list[int]) -> list[list[int]]:
    """Build face stacking data from the current folded layer graph."""
    plane_layer_pos: dict[int, int] = {}
    try:
        for gen_idx, generation in enumerate(nx.topological_generations(origami.layers.reverse())):
            for pid in generation:
                plane_layer_pos[pid] = gen_idx
    except Exception:
        return []

    face_orders: list[list[int]] = []
    for i, fid_i in enumerate(sorted_fids):
        pid_i = origami.faces[fid_i].plane_id
        li = plane_layer_pos.get(pid_i)
        for j, fid_j in enumerate(sorted_fids):
            if j <= i:
                continue
            pid_j = origami.faces[fid_j].plane_id
            if pid_i == pid_j:
                continue
            lj = plane_layer_pos.get(pid_j)
            if li is None or lj is None or li == lj:
                continue
            face_orders.append([i, j, 1 if lj > li else -1])
    return face_orders


def origami_to_fold(origami: OrigamiCore) -> dict:
    """Build a FOLD dict for the folded (or current) state of *origami*."""
    sorted_vids = sorted(origami.vertices)
    vid_to_idx = {vid: i for i, vid in enumerate(sorted_vids)}

    vertices_coords = [
        [_norm_coord(c) for c in origami.vertices[vid].pos]
        for vid in sorted_vids
    ]

    sorted_fids = sorted(origami.faces)
    faces_vertices = [
        [vid_to_idx[vid] for vid in origami._face_vids(fid)]
        for fid in sorted_fids
    ]

    seen_edges: set[tuple[int, int]] = set()
    edges_vertices: list[list[int]] = []
    edges_assignment: list[str] = []
    edges_foldAngle: list[int] = []

    for edge_key in sorted(origami.half_edges):
        v1, v2 = edge_key
        norm = (min(v1, v2), max(v1, v2))
        if norm in seen_edges:
            continue
        seen_edges.add(norm)

        he = origami.half_edges[edge_key]
        etype = he.type
        if etype is None and he.twin is not None:
            etype = he.twin.type
        if etype is None:
            etype = "U"

        edges_vertices.append([vid_to_idx[norm[0]], vid_to_idx[norm[1]]])
        edges_assignment.append(etype)
        edges_foldAngle.append(_FOLD_ANGLE.get(etype, 0))

    fold_dict = {
        "file_spec": 1,
        "file_creator": "OrigamiLib",
        "file_classes": ["foldedForm"],
        "vertices_coords": vertices_coords,
        "faces_vertices": faces_vertices,
        "edges_vertices": edges_vertices,
        "edges_assignment": edges_assignment,
        "edges_foldAngle": edges_foldAngle,
        "faceOrders": _get_face_orders(origami, sorted_fids),
    }

    return fold_dict


def _get_plane_relation_orders(origami: OrigamiCore, sorted_fids: list[int]) -> list[list[int]]:
    """Return [i, j, 0] for every pair of faces that share the same plane."""
    plane_relation_orders: list[list[int]] = []
    for i, fid_i in enumerate(sorted_fids):
        pid_i = origami.faces[fid_i].plane_id
        if pid_i is None:
            continue
        for j, fid_j in enumerate(sorted_fids):
            if j <= i:
                continue
            pid_j = origami.faces[fid_j].plane_id
            if pid_i == pid_j:
                plane_relation_orders.append([i, j, 0])
    return plane_relation_orders


def cp_to_fold(origami: OrigamiCore) -> dict:
    """Build a FOLD dict for the unfolded crease pattern of *origami*, always
    including fold angles and face/plane stacking orders."""
    crease_pattern = CreasePattern.from_origami(origami, unfolded=True)

    sorted_vids = sorted(crease_pattern.vertices)
    vid_to_idx = {vid: i for i, vid in enumerate(sorted_vids)}

    vertices_coords = [
        [_norm_coord(crease_pattern.vertices[vid][0]), _norm_coord(crease_pattern.vertices[vid][1])]
        for vid in sorted_vids
    ]

    sorted_fids = sorted(origami.faces)
    faces_vertices = [
        [vid_to_idx[vid] for vid in origami._face_vids(fid) if vid in vid_to_idx]
        for fid in sorted_fids
    ]

    seen_edges: set[tuple[int, int]] = set()
    edges_vertices: list[list[int]] = []
    edges_assignment: list[str] = []
    for segment in crease_pattern.segments:
        v1_id, v2_id = sorted(segment.vertex_ids)
        key = (v1_id, v2_id)
        if key in seen_edges:
            continue
        if v1_id not in vid_to_idx or v2_id not in vid_to_idx:
            continue
        seen_edges.add(key)
        edges_vertices.append([vid_to_idx[v1_id], vid_to_idx[v2_id]])
        edges_assignment.append(segment.edge_type)

    fold_dict = {
        "file_spec": 1,
        "file_creator": "OrigamiLib",
        "file_classes": ["creasePattern"],
        "frame_attributes": ["2D"],
        "vertices_coords": vertices_coords,
        "faces_vertices": faces_vertices,
        "edges_vertices": edges_vertices,
        "edges_assignment": edges_assignment,
        "edges_foldAngle": [
            _FOLD_ANGLE.get(edge_type, 0)
            for edge_type in edges_assignment
        ],
        "faceOrders": (
            _get_face_orders(origami, sorted_fids)
            + _get_plane_relation_orders(origami, sorted_fids)
        ),
    }

    return fold_dict




def fold_to_json(fold_dict: dict, indent: int | None = 2) -> str:
    return json.dumps(fold_dict, indent=indent)


# ---------------------------------------------------------------------------
# FOLD → Origami import
# ---------------------------------------------------------------------------

def is_fold_format(data: dict | str) -> bool:
    """Return True if *data* looks like a FOLD-format dict or JSON string."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return False
    return isinstance(data, dict) and "vertices_coords" in data


def fold_to_origami(fold_data: dict | str, origami_factory: Callable[[], OrigamiCore]) -> OrigamiCore:
    """Reconstruct an Origami object from a FOLD-format dict or JSON string.

    Vertex and face indices in FOLD are 0-based; they are shifted to 1-based
    internally to match the rest of the library.
    """
    from .representation import OrigamiRepresentation  # local to avoid circular import at module level

    if isinstance(fold_data, str):
        fold_data = json.loads(fold_data)

    vertices_coords: list = fold_data.get("vertices_coords", [])
    faces_verts_0: list = fold_data.get("faces_vertices", [])
    edges_verts_0: list = fold_data.get("edges_vertices", [])
    edge_assignments: list = fold_data.get("edges_assignment", [])
    face_orders_raw: list = fold_data.get("faceOrders", [])
    file_classes: list = fold_data.get("file_classes", [])

    is_cp = "creasePattern" in file_classes

    # ------------------------------------------------------------------
    # 1. Vertices  (FOLD index i  →  internal ID  i+1)
    # ------------------------------------------------------------------
    vertices: dict[int, Vertex] = {}
    for i, coords in enumerate(vertices_coords):
        vid = i + 1
        vertices[vid] = Vertex(id=vid, pos=np.array([float(c) for c in coords], dtype=float))

    # ------------------------------------------------------------------
    # 2. Edge-type lookup  (1-indexed normalised key)
    # ------------------------------------------------------------------
    edge_type_map: dict[tuple[int, int], str] = {}
    for (v1_0, v2_0), etype in zip(edges_verts_0, edge_assignments):
        v1, v2 = v1_0 + 1, v2_0 + 1
        edge_type_map[(min(v1, v2), max(v1, v2))] = etype

    # ------------------------------------------------------------------
    # 3. Face cycles  (1-indexed vertex lists)
    # ------------------------------------------------------------------
    face_cycles: dict[int, list[int]] = {}
    for i, fv0 in enumerate(faces_verts_0):
        face_cycles[i + 1] = [v + 1 for v in fv0]

    # ------------------------------------------------------------------
    # 4. Face orientations
    #    If explicitly provided in FOLD, use them by face index.
    #    Otherwise, default every face orientation to 0.
    # ------------------------------------------------------------------
    raw_face_orientations = fold_data.get("faceOrientations")
    orientations: dict[int, int] = {}
    if raw_face_orientations is not None:
        if not isinstance(raw_face_orientations, list):
            raise ValueError("FOLD field 'faceOrientations' must be a list when provided.")
        if len(raw_face_orientations) != len(face_cycles):
            raise ValueError(
                "FOLD field 'faceOrientations' must have one entry per face."
            )
        for index, raw_orientation in enumerate(raw_face_orientations, start=1):
            orientations[index] = int(raw_orientation)
    else:
        for fid in face_cycles:
            orientations[fid] = 0

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fid, cycle in face_cycles.items():
        n = len(cycle)
        for k in range(n):
            v1, v2 = cycle[k], cycle[(k + 1) % n]
            edge_to_faces[(min(v1, v2), max(v1, v2))].append(fid)

    # ------------------------------------------------------------------
    # 5. Plane grouping
    #    CP: all faces share one plane.
    #    Folded form: connected components via flat (F) edges.
    # ------------------------------------------------------------------
    parent: dict[int, int] = {fid: fid for fid in face_cycles}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        parent[_find(x)] = _find(y)

    if is_cp:
        for fid in face_cycles:
            _union(fid, min(face_cycles))
    else:
        for norm_e, fids in edge_to_faces.items():
            if len(fids) == 2:
                f1, f2 = fids
                if edge_type_map.get(norm_e, "F") == "F":
                    _union(f1, f2)

    root_to_pid: dict[int, int] = {}
    next_pid = 1
    face_to_pid: dict[int, int] = {}
    for fid in sorted(face_cycles):
        root = _find(fid)
        if root not in root_to_pid:
            root_to_pid[root] = next_pid
            next_pid += 1
        face_to_pid[fid] = root_to_pid[root]

    planes: dict[int, Plane] = {pid: Plane(id=pid, face_ids=[]) for pid in range(1, next_pid)}
    for fid, pid in face_to_pid.items():
        planes[pid].face_ids.append(fid)

    # ------------------------------------------------------------------
    # 6. Layer DAG from faceOrders
    #    [i, j, 1]  → face j above face i  → plane(j) above plane(i)
    #    [i, j, -1] → face j below face i  → plane(i) above plane(j)
    # ------------------------------------------------------------------
    layers: nx.DiGraph = nx.DiGraph()
    for pid in planes:
        layers.add_node(pid)

    for triple in face_orders_raw:
        if len(triple) != 3:
            continue
        fi_0, fj_0, k = triple
        fi, fj = fi_0 + 1, fj_0 + 1
        pid_i = face_to_pid.get(fi)
        pid_j = face_to_pid.get(fj)
        if pid_i is None or pid_j is None or pid_i == pid_j:
            continue
        if k == 1:
            layers.add_edge(pid_j, pid_i)   # j above i
        elif k == -1:
            layers.add_edge(pid_i, pid_j)   # i above j

    try:
        if nx.is_directed_acyclic_graph(layers):
            reduced = nx.transitive_reduction(layers)
            for pid in planes:              # transitive_reduction preserves nodes, but be safe
                reduced.add_node(pid)
            layers = reduced
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 7. Assemble the origami object
    # ------------------------------------------------------------------
    origami = origami_factory()
    origami.vertices = vertices
    origami.faces = {
        fid: Face(id=fid, orientation=orientations[fid], plane_id=face_to_pid[fid])
        for fid in sorted(face_cycles)
    }
    origami.planes = planes
    origami.layers = layers
    origami.half_edges = {}

    prev_by_half_edge: dict = {}
    for fid in sorted(face_cycles):
        face = origami.faces[fid]
        cycle = face_cycles[fid]
        n = len(cycle)
        cycle_hes: list[HalfEdge] = []
        for idx in range(n):
            origin_vid = cycle[idx]
            next_vid = cycle[(idx + 1) % n]
            he = HalfEdge()
            he.origin = origami.vertices[origin_vid]
            he.face = face
            cycle_hes.append(he)
            origami.half_edges[(origin_vid, next_vid)] = he
        for idx, he in enumerate(cycle_hes):
            he.next = cycle_hes[(idx + 1) % n]
            prev_by_half_edge[he] = cycle_hes[(idx - 1) % n]
        face.edge = cycle_hes[0]

    # Link twins and assign edge types; create boundary half-edges
    boundary_hes: dict = {}
    for edge_key, he in list(origami.half_edges.items()):
        if he.twin is not None:
            continue
        v1, v2 = edge_key
        norm = (min(v1, v2), max(v1, v2))
        etype = edge_type_map.get(norm, "B")

        twin_key = (v2, v1)
        twin = origami.half_edges.get(twin_key)
        if twin is not None:
            he.twin = twin
            twin.twin = he
            he.type = etype
            twin.type = etype
        else:
            bhe = HalfEdge("B")
            bhe.origin = origami.vertices[v2]
            bhe.face = None
            bhe.twin = he
            he.twin = bhe
            he.type = "B"
            boundary_hes[twin_key] = bhe

    origami.half_edges.update(boundary_hes)
    OrigamiRepresentation._build_boundary_next_map(origami, prev_by_half_edge)
    OrigamiRepresentation._assign_vertex_half_edges(origami)

    return origami
