from __future__ import annotations

import math
import numpy as np
import networkx as nx
from .config import EPS

Line = tuple[float, float, float]  # A*x + B*y + C = 0


def find_linear_paths_nx(
    vertices: dict[int, np.ndarray],
    edges: list[tuple[int, int]],
    s_id: int,
    t_id: int,
    tolerance: float = EPS,
) -> list[list[int]]:
    # 1. Create the Graph
    G = nx.Graph() # Use nx.DiGraph() if the edges have a specific direction
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    # 2. Find all simple paths from s to t
    all_paths = list(nx.all_simple_paths(G, source=s_id, target=t_id))

    # 3. Check each path vertex for collinearity with S-T
    def as_point(node_id: int) -> np.ndarray:
        node_value = vertices[node_id]
        if hasattr(node_value, "pos"):
            node_value = node_value.pos
        point = np.asarray(node_value, dtype=float)
        if point.ndim != 1 or point.size < 2:
            raise ValueError(f"Vertex {node_id} does not contain a valid 2D position.")
        return point[:2]

    S = as_point(s_id)
    T = as_point(t_id)
    ST_vec = T - S

    def is_on_line(node_id: int) -> bool:
        P = as_point(node_id)
        if np.array_equal(P, S) or np.array_equal(P, T):
            return True

        SP_vec = P - S

        # Cross product in 2D: if 0, points are collinear
        # cross = x1*y2 - y1*x2
        cross_prod = np.cross(SP_vec, ST_vec)
        if abs(cross_prod) > tolerance:
            return False

        # Dot product check to ensure P is BETWEEN S and T (not behind S or past T)
        # 0 <= dot(SP, ST) <= dot(ST, ST)
        dot_sp_st = np.dot(SP_vec, ST_vec)
        dot_st_st = np.dot(ST_vec, ST_vec)

        return 0 <= dot_sp_st <= dot_st_st

    # 4. Filter paths where all vertices are linear
    linear_paths = [path for path in all_paths if all(is_on_line(n) for n in path)]

    return linear_paths


def get_canonical_edge_direction(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (p1, p2) in canonical direction:
    - prefer bottom-to-top (increasing y)
    - if horizontal, prefer left-to-right (increasing x)
    This makes +1 always mean 'left of the directed line',
    which visually means: left of vertical lines, above horizontal lines.
    """

    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]

    if abs(dy) > EPS:
        # prefer bottom to top
        if dy > 0:
            return p1, p2
        else:
            return p2, p1
    else:
        # horizontal: prefer left to right
        if dx > 0:
            return p1, p2
        else:
            return p2, p1

def get_line_equation(p1: np.ndarray, p2: np.ndarray) -> Line:
    # Build a normalized line equation A*x + B*y + C = 0.
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    A = y2 - y1
    B = x1 - x2
    C = -A * x1 - B * y1
    norm = math.hypot(A, B)
    if norm > 0:
        A /= norm
        B /= norm
        C /= norm
        # Cartesian-oriented sign convention:
        # - prefer +y as positive side (B > 0)
        # - for vertical lines (B ~= 0), prefer +x (A > 0)
        if B < -EPS or (abs(B) <= EPS and A < 0):
            A, B, C = -A, -B, -C
    else:
        A, B, C = 0.0, 0.0, 0.0
    return A, B, C

def sign(x: float) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def point_side_to_line(point: np.ndarray, line: Line) -> int:
    A, B, C = line
    d = A * point[0] + B * point[1] + C
    if abs(d) < EPS:
        return 0
    return sign(d)

def segment_line_intersection(line: Line, segment: tuple[np.ndarray, np.ndarray]) -> float | None:
    p1, p2 = segment
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    A, B, C = line
    D = A * dx + B * dy
    if math.fabs(D) < EPS:
        return None
    N = - (A * x1 + B * y1 + C)
    t = N / D
    segment_length = math.hypot(dx, dy)
    # Snap near-endpoint intersections by absolute distance, not ratio. This
    # avoids creating sliver edges when the crossed segment is very short.
    if abs(t) * segment_length < EPS:
        return 0
    if abs(1 - t) * segment_length < EPS:
        return 1
    return t

def reflect_point(point: np.ndarray, line_eq: Line) -> np.ndarray:
    p_x, p_y = point
    A, B, C = line_eq
    denominator = A**2 + B**2
    if denominator == 0: return np.array([p_x, p_y])
    t = -2 * (A * p_x + B * p_y + C) / denominator
    return np.array([p_x + A * t, p_y + B * t])

def get_overlap(min1: float, max1: float, min2: float, max2: float, eps: float = 1e-9) -> bool:
    return max(min1, min2) < min(max1, max2) - eps

def project_on_axis(points: list[np.ndarray], axis: tuple[float, float]) -> tuple[float, float]:
    dot_products = []

    for x, y in points:
        dot_products.append(x * axis[0] + y * axis[1])

    return min(dot_products), max(dot_products)


def do_faces_overlap(face1: list[np.ndarray], face2: list[np.ndarray]) -> bool:

    all_faces = [face1, face2]

    for points in all_faces:

        num_vertices = len(points)

        for i in range(num_vertices):
            p1 = points[i]
            p2 = points[(i + 1) % num_vertices]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            axis = (-dy, dx)

            min1, max1 = project_on_axis(face1, axis)
            min2, max2 = project_on_axis(face2, axis)

            if not get_overlap(min1, max1, min2, max2):
                return False

    return True
