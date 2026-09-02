from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from .config import EPS
from .plotly_support import resolve_plotly_renderer
from .raster import RasterViewport, create_canvas, draw_line
from .utils import get_line_equation

if TYPE_CHECKING:
    from .core import OrigamiCore

PointLike = tuple[float, float] | np.ndarray
FaceTransforms = dict[int, tuple[np.ndarray, np.ndarray]]

PLOT_MARGINS = dict(l=20, r=20, t=40, b=20)
PLOT_WIDTH_PX = 560  # fixed "medium" figure size
PLOT_HEIGHT_PX = 580
IMAGE_SCALE = 2.0
EDGE_STYLES = {
    "B": {"color": "black", "dash": "solid", "width": 2.0},
    "F": {"color": "dimgray", "dash": "solid", "width": 2.0},
    "M": {"color": "red", "dash": "solid", "width": 2.0},
    "V": {"color": "blue", "dash": "solid", "width": 2.0},
}

@dataclass(frozen=True)
class CreaseSegment:
    edge_type: str
    vertex_ids: tuple[int, ...]
    points: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class VertexFoldabilityCheck:
    """Local flat-foldability diagnostics for one crease-map vertex."""

    vertex_id: int | None
    point: tuple[float, float]
    degree: int
    is_boundary: bool
    sector_angles: tuple[float, ...]
    assignments: tuple[str, ...]
    kawasaki_residual: float | None = None
    maekawa_difference: int | None = None
    issues: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class CreasePatternValidationResult:
    """Structured result for crease pattern validation."""

    is_valid: bool
    is_planar: bool
    satisfies_local_flat_foldability: bool
    has_complete_mv_assignment: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    vertex_checks: tuple[VertexFoldabilityCheck, ...]

    @property
    def invalid_vertex_ids(self) -> tuple[int, ...]:
        return tuple(
            check.vertex_id
            for check in self.vertex_checks
            if check.vertex_id is not None and not check.is_valid
        )


class CreasePattern:
    """A flattened crease pattern view extracted from an origami model."""

    def __init__(self, segments: list[CreaseSegment], vertices: dict[int, tuple[float, float]]) -> None:
        self.segments = segments
        self.vertices = vertices

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vertices={len(self.vertices)}, segments={len(self.segments)})"
        )

    @classmethod
    def from_origami(cls, origami: OrigamiCore, unfolded: bool = True) -> "CreasePattern":
        face_transforms = (
            cls._get_unfolded_face_transforms(origami)
            if unfolded
            else cls._get_identity_face_transforms(origami)
        )
        vertices = {
            vid: tuple(
                float(value)
                for value in cls._apply_vertex_transform(vid, origami, face_transforms)
            )
            for vid in sorted(origami.vertices)
        }
        segment_records: dict[
            tuple[tuple[float, float], tuple[float, float]],
            dict[str, object],
        ] = {}

        for edge, half_edge in origami.half_edges.items():
            if half_edge.face is None:
                continue
            if half_edge.twin is None or half_edge.origin is None or half_edge.twin.origin is None:
                continue

            p1 = vertices[half_edge.origin.id]
            p2 = vertices[half_edge.twin.origin.id]
            key = cls._segment_key(p1, p2)
            exact_points = cls._ordered_segment_points(p1, p2)

            record = segment_records.setdefault(
                key,
                {
                    "types": set(),
                    # One id set per endpoint in `points` order, so that
                    # distinct vertices which land on the same point (e.g.
                    # overlapping layers in a folded/identity projection)
                    # collapse to a single representative id per point
                    # instead of an ambiguous flat set of every id involved.
                    "vertex_ids_by_point": (set(), set()),
                    "points": exact_points,
                },
            )
            record["types"].add(half_edge.type)
            if half_edge.twin is not None:
                record["types"].add(half_edge.twin.type)

            origin_point = cls._point_key(p1)
            point_a, point_b = record["points"]
            ids_a, ids_b = record["vertex_ids_by_point"]
            if origin_point == point_a:
                ids_a.add(half_edge.origin.id)
                ids_b.add(half_edge.twin.origin.id)
            else:
                ids_b.add(half_edge.origin.id)
                ids_a.add(half_edge.twin.origin.id)

        segments = [
            CreaseSegment(
                edge_type=cls._resolve_edge_type(record["types"]),
                vertex_ids=(min(record["vertex_ids_by_point"][0]), min(record["vertex_ids_by_point"][1])),
                points=record["points"],
            )
            for key, record in sorted(segment_records.items())
        ]
        if unfolded:
            vertices, segments = cls._align_to_reference_square(origami, vertices, segments)
        return cls(segments=segments, vertices=vertices)

    @classmethod
    def _get_identity_face_transforms(cls, origami: OrigamiCore) -> FaceTransforms:
        identity = np.eye(2, dtype=float)
        zero = np.zeros(2, dtype=float)
        return {
            fid: (identity.copy(), zero.copy())
            for fid in origami.faces
        }

    @classmethod
    def _get_unfolded_face_transforms(cls, origami: OrigamiCore) -> FaceTransforms:
        if not origami.faces:
            return {}

        transforms = {}
        start_fid = next(iter(sorted(origami.faces)))
        transforms[start_fid] = (np.eye(2, dtype=float), np.zeros(2, dtype=float))
        queue = [start_fid]

        while queue:
            fid = queue.pop(0)
            parent_matrix, parent_offset = transforms[fid]
            face = origami.faces[fid]

            for half_edge in origami._walk_face_half_edges(face):
                neighbor = half_edge.twin.face if half_edge.twin is not None else None
                if neighbor is None or neighbor.id in transforms:
                    continue

                should_reflect = (
                    half_edge.type != "F"
                    or face.orientation != neighbor.orientation
                )

                if not should_reflect:
                    transforms[neighbor.id] = (parent_matrix.copy(), parent_offset.copy())
                else:
                    p1 = cls._apply_transform(parent_matrix, parent_offset, half_edge.origin.pos)
                    p2 = cls._apply_transform(parent_matrix, parent_offset, half_edge.twin.origin.pos)
                    reflection_matrix, reflection_offset = cls._reflection_transform(p1, p2)
                    transforms[neighbor.id] = (
                        reflection_matrix @ parent_matrix,
                        reflection_matrix @ parent_offset + reflection_offset,
                    )
                queue.append(neighbor.id)

        if len(transforms) != len(origami.faces):
            missing_fids = set(origami.faces) - set(transforms)
            raise ValueError(
                "Could not flatten all faces into a crease map. "
                f"Disconnected face set: {sorted(missing_fids)}"
            )

        return transforms

    @staticmethod
    def _apply_transform(matrix: np.ndarray, offset: np.ndarray, point: PointLike) -> np.ndarray:
        point_xy = np.asarray(point, dtype=float)[:2]
        return matrix @ point_xy + offset

    @classmethod
    def _apply_vertex_transform(cls, vid: int, origami: OrigamiCore, face_transforms: FaceTransforms) -> np.ndarray:
        vertex = origami.vertices[vid]
        best_fid = None
        for half_edge in origami.half_edges.values():
            if half_edge.face is None or half_edge.origin is None or half_edge.origin.id != vid:
                continue
            fid = half_edge.face.id
            if best_fid is None or fid < best_fid:
                best_fid = fid
        if best_fid is not None:
            matrix, offset = face_transforms[best_fid]
            return cls._apply_transform(matrix, offset, vertex.pos)
        return np.asarray(vertex.pos, dtype=float)[:2]

    @staticmethod
    def _reflection_transform(p1: PointLike, p2: PointLike) -> tuple[np.ndarray, np.ndarray]:
        line_eq = get_line_equation(p1, p2)
        a, b, c = line_eq
        normal = np.array([a, b], dtype=float)
        reflection_matrix = np.eye(2, dtype=float) - 2.0 * np.outer(normal, normal)
        reflection_offset = -2.0 * c * normal
        return reflection_matrix, reflection_offset

    @classmethod
    def _align_to_reference_square(
        cls,
        origami: OrigamiCore,
        vertices: dict[int, tuple[float, float]],
        segments: list[CreaseSegment],
    ) -> tuple[dict[int, tuple[float, float]], list[CreaseSegment]]:
        reference_vertices = {
            1: np.array([0.0, 1.0], dtype=float),
            2: np.array([1.0, 1.0], dtype=float),
            3: np.array([1.0, 0.0], dtype=float),
            4: np.array([0.0, 0.0], dtype=float),
        }
        if not all(vid in vertices for vid in (1, 2, 4)):
            return vertices, segments

        try:
            alignment_matrix, alignment_offset = cls._corner_affine_transform(
                vertices=vertices,
                reference_vertices=reference_vertices,
            )
        except np.linalg.LinAlgError:
            available_corner_ids = [vid for vid in reference_vertices if vid in vertices]
            source_points = np.array([vertices[vid] for vid in available_corner_ids], dtype=float)
            target_points = np.array(
                [reference_vertices[vid] for vid in available_corner_ids],
                dtype=float,
            )
            alignment_matrix, alignment_offset = cls._best_fit_rigid_transform(source_points, target_points)

        aligned_vertices = {
            vid: cls._stabilize_point(
                cls._apply_transform(alignment_matrix, alignment_offset, point)
            )
            for vid, point in vertices.items()
        }
        aligned_segments = [
            CreaseSegment(
                edge_type=segment.edge_type,
                vertex_ids=segment.vertex_ids,
                points=cls._ordered_segment_points(
                    cls._stabilize_point(
                        cls._apply_transform(alignment_matrix, alignment_offset, segment.points[0])
                    ),
                    cls._stabilize_point(
                        cls._apply_transform(alignment_matrix, alignment_offset, segment.points[1])
                    ),
                ),
            )
            for segment in segments
        ]
        return aligned_vertices, aligned_segments

    @staticmethod
    def _corner_affine_transform(
        vertices: dict[int, tuple[float, float]],
        reference_vertices: dict[int, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        source_origin = np.asarray(vertices[4], dtype=float)
        source_basis = np.column_stack((
            np.asarray(vertices[2], dtype=float) - source_origin,
            np.asarray(vertices[1], dtype=float) - source_origin,
        ))

        target_origin = reference_vertices[4]
        target_basis = np.column_stack((
            reference_vertices[2] - target_origin,
            reference_vertices[1] - target_origin,
        ))

        alignment_matrix = target_basis @ np.linalg.inv(source_basis)
        alignment_offset = target_origin - alignment_matrix @ source_origin
        return alignment_matrix, alignment_offset

    @staticmethod
    def _best_fit_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source_center = np.mean(source_points, axis=0)
        target_center = np.mean(target_points, axis=0)
        source_shifted = source_points - source_center
        target_shifted = target_points - target_center
        covariance = source_shifted.T @ target_shifted
        u, _, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = vt.T @ u.T
        translation = target_center - rotation @ source_center
        return rotation, translation

    @staticmethod
    def _stabilize_scalar(value: float, decimals: int = 12) -> float:
        stabilized = float(round(float(value), decimals))
        return 0.0 if stabilized == -0.0 else stabilized

    @classmethod
    def _stabilize_point(cls, point: PointLike) -> tuple[float, float]:
        return (
            cls._stabilize_scalar(point[0]),
            cls._stabilize_scalar(point[1]),
        )

    @staticmethod
    def _snap_scalar(value: float, grid_size: float = EPS) -> float:
        return float(round(float(value) / grid_size) * grid_size)

    @classmethod
    def _snap_point(cls, point: PointLike, grid_size: float = EPS) -> tuple[float, float]:
        return (
            cls._snap_scalar(point[0], grid_size=grid_size),
            cls._snap_scalar(point[1], grid_size=grid_size),
        )

    @staticmethod
    def _segment_key(
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        snapped_p1 = CreasePattern._snap_point(p1)
        snapped_p2 = CreasePattern._snap_point(p2)
        return (snapped_p1, snapped_p2) if snapped_p1 <= snapped_p2 else (snapped_p2, snapped_p1)

    @staticmethod
    def _resolve_edge_type(edge_types: set[str]) -> str:
        normalized_types = {edge_type for edge_type in edge_types if edge_type is not None}
        for preferred in ("M", "V", "F", "B"):
            if preferred in normalized_types:
                return preferred
        return "F"

    @staticmethod
    def _cross_2d(u: PointLike, v: PointLike) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    @staticmethod
    def _point_distance(p1: PointLike, p2: PointLike) -> float:
        return float(math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])))

    @classmethod
    def _points_close(cls, p1: PointLike, p2: PointLike, tolerance: float = EPS) -> bool:
        return cls._point_distance(p1, p2) <= tolerance

    @staticmethod
    def _point_key(point: PointLike) -> tuple[float, float]:
        return (float(point[0]), float(point[1]))

    @classmethod
    def _ordered_segment_points(cls, p1: PointLike, p2: PointLike) -> tuple[tuple[float, float], tuple[float, float]]:
        point1 = cls._point_key(p1)
        point2 = cls._point_key(p2)
        return (point1, point2) if point1 <= point2 else (point2, point1)

    @classmethod
    def _canonicalize_points(cls, points: Iterable[PointLike], tolerance: float = EPS) -> dict[tuple[float, float], tuple[float, float]]:
        point_aliases = {}
        representatives: list[tuple[float, float]] = []
        for point in sorted({cls._point_key(point) for point in points}):
            representative = next(
                (
                    existing
                    for existing in representatives
                    if cls._points_close(point, existing, tolerance=tolerance)
                ),
                None,
            )
            if representative is None:
                representative = point
                representatives.append(point)
            point_aliases[point] = representative
        return point_aliases

    @classmethod
    def _resolve_point_alias(
        cls,
        point: PointLike,
        point_aliases: dict[tuple[float, float], tuple[float, float]] | None = None,
    ) -> tuple[float, float]:
        point_key = cls._point_key(point)
        if point_aliases is None:
            return point_key
        return point_aliases.get(point_key, point_key)

    @classmethod
    def _format_point(cls, point: PointLike) -> str:
        snapped = cls._snap_point(point)
        return f"({snapped[0]:.6g}, {snapped[1]:.6g})"

    @classmethod
    def _format_segment(cls, segment: CreaseSegment) -> str:
        p1, p2 = segment.points
        return f"{cls._format_point(p1)} -> {cls._format_point(p2)} [{segment.edge_type}]"

    @classmethod
    def _point_on_segment(
        cls,
        point: PointLike,
        start: PointLike,
        end: PointLike,
        tolerance: float = EPS,
        strict: bool = False,
    ) -> bool:
        point_arr = np.asarray(point, dtype=float)
        start_arr = np.asarray(start, dtype=float)
        end_arr = np.asarray(end, dtype=float)
        segment = end_arr - start_arr
        segment_length = np.linalg.norm(segment)
        if segment_length <= tolerance:
            return np.linalg.norm(point_arr - start_arr) <= tolerance

        offset = point_arr - start_arr
        distance_to_line = abs(cls._cross_2d(segment, offset)) / segment_length
        if distance_to_line > tolerance:
            return False

        projection = float(np.dot(offset, segment) / np.dot(segment, segment))
        if strict:
            return tolerance < projection < 1.0 - tolerance
        return -tolerance <= projection <= 1.0 + tolerance

    @classmethod
    def _segments_match(
        cls,
        p1: PointLike,
        p2: PointLike,
        q1: PointLike,
        q2: PointLike,
        tolerance: float = EPS,
    ) -> bool:
        return (
            cls._points_close(p1, q1, tolerance) and cls._points_close(p2, q2, tolerance)
        ) or (
            cls._points_close(p1, q2, tolerance) and cls._points_close(p2, q1, tolerance)
        )

    @classmethod
    def _segments_are_collinear(cls, p1: PointLike, p2: PointLike, q1: PointLike, q2: PointLike, tolerance: float = EPS) -> bool:
        return cls._point_on_segment(q1, p1, p2, tolerance=tolerance) and cls._point_on_segment(
            q2, p1, p2, tolerance=tolerance
        )

    @classmethod
    def _collinear_overlap_length(cls, p1: PointLike, p2: PointLike, q1: PointLike, q2: PointLike) -> float:
        axis = 0 if abs(float(p2[0]) - float(p1[0])) >= abs(float(p2[1]) - float(p1[1])) else 1
        p_start, p_end = sorted((float(p1[axis]), float(p2[axis])))
        q_start, q_end = sorted((float(q1[axis]), float(q2[axis])))
        return max(0.0, min(p_end, q_end) - max(p_start, q_start))

    @classmethod
    def _segments_properly_intersect(cls, p1: PointLike, p2: PointLike, q1: PointLike, q2: PointLike, tolerance: float = EPS) -> bool:
        p1_arr = np.asarray(p1, dtype=float)
        p2_arr = np.asarray(p2, dtype=float)
        q1_arr = np.asarray(q1, dtype=float)
        q2_arr = np.asarray(q2, dtype=float)
        o1 = cls._cross_2d(p2_arr - p1_arr, q1_arr - p1_arr)
        o2 = cls._cross_2d(p2_arr - p1_arr, q2_arr - p1_arr)
        o3 = cls._cross_2d(q2_arr - q1_arr, p1_arr - q1_arr)
        o4 = cls._cross_2d(q2_arr - q1_arr, p2_arr - q1_arr)
        return (
            ((o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance))
            and ((o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance))
        )

    @classmethod
    def _segment_pair_issue(
        cls,
        first: CreaseSegment,
        second: CreaseSegment,
        point_tolerance: float = EPS,
    ) -> str | None:
        p1, p2 = first.points
        q1, q2 = second.points

        if cls._segments_match(p1, p2, q1, q2, tolerance=point_tolerance):
            return (
                "Duplicate segments detected: "
                f"{cls._format_segment(first)} and {cls._format_segment(second)}."
            )

        if cls._segments_are_collinear(p1, p2, q1, q2, tolerance=point_tolerance):
            overlap_length = cls._collinear_overlap_length(p1, p2, q1, q2)
            if overlap_length > point_tolerance:
                return (
                    "Collinear segments overlap without an intermediate split vertex: "
                    f"{cls._format_segment(first)} and {cls._format_segment(second)}."
                )
            return None

        if cls._segments_properly_intersect(p1, p2, q1, q2, tolerance=point_tolerance):
            return (
                "Segments cross away from a shared vertex: "
                f"{cls._format_segment(first)} and {cls._format_segment(second)}."
            )

        for point in (p1, p2):
            if cls._point_on_segment(point, q1, q2, tolerance=point_tolerance, strict=True):
                return (
                    "A segment endpoint lies on another segment without a split vertex: "
                    f"{cls._format_segment(first)} and {cls._format_segment(second)}."
                )

        for point in (q1, q2):
            if cls._point_on_segment(point, p1, p2, tolerance=point_tolerance, strict=True):
                return (
                    "A segment endpoint lies on another segment without a split vertex: "
                    f"{cls._format_segment(first)} and {cls._format_segment(second)}."
                )

        return None

    @classmethod
    def _validate_boundary_cycle(
        cls,
        boundary_segments: list[CreaseSegment],
        point_aliases: dict[tuple[float, float], tuple[float, float]] | None = None,
    ) -> list[str]:
        if not boundary_segments:
            return []

        boundary_graph = defaultdict(set)
        for segment in boundary_segments:
            p1 = cls._resolve_point_alias(segment.points[0], point_aliases)
            p2 = cls._resolve_point_alias(segment.points[1], point_aliases)
            boundary_graph[p1].add(p2)
            boundary_graph[p2].add(p1)

        issues = []
        for point, neighbors in sorted(boundary_graph.items()):
            if len(neighbors) != 2:
                issues.append(
                    "Boundary edges must form a single simple cycle; "
                    f"boundary vertex {cls._format_point(point)} has {len(neighbors)} incident boundary edges."
                )

        if issues:
            return issues

        visited = set()
        stack = [next(iter(boundary_graph))]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(neighbor for neighbor in boundary_graph[current] if neighbor not in visited)

        if len(visited) != len(boundary_graph):
            issues.append("Boundary edges do not form one connected closed cycle.")
        elif len(boundary_segments) != len(boundary_graph):
            issues.append("Boundary edges do not form a simple cycle.")

        return issues

    @classmethod
    def _vertex_lookup(
        cls,
        vertices: dict[int, tuple[float, float]],
        point_aliases: dict[tuple[float, float], tuple[float, float]] | None = None,
    ) -> dict[tuple[float, float], int]:
        lookup = {}
        for vid in sorted(vertices):
            lookup.setdefault(cls._resolve_point_alias(vertices[vid], point_aliases), vid)
        return lookup

    def _build_incidence(
        self,
        point_aliases: dict[tuple[float, float], tuple[float, float]] | None = None,
    ) -> dict[tuple[float, float], list[tuple[CreaseSegment, tuple[float, float]]]]:
        incidence = defaultdict(list)
        for segment in self.segments:
            p1 = self._resolve_point_alias(segment.points[0], point_aliases)
            p2 = self._resolve_point_alias(segment.points[1], point_aliases)
            incidence[p1].append((segment, p2))
            incidence[p2].append((segment, p1))
        return incidence

    def validate_flat_foldability(
        self,
        angle_tolerance: float = 1.5e-2,
        point_tolerance: float = EPS,
        require_mv_assignment: bool = False,
    ) -> CreasePatternValidationResult:
        """Run practical flat-foldability checks on the crease pattern.

        The validator applies standard necessary checks:
        - planar embedding sanity for the segment graph,
        - boundary-cycle sanity when boundary edges are present,
        - Kawasaki's theorem at interior vertices,
        - Maekawa's theorem where every incident crease is assigned ``M`` or ``V``.

        A passing result rules out many invalid crease patterns, but it does not
        prove global flat foldability for arbitrary crease patterns.
        """

        all_points = list(self.vertices.values())
        for segment in self.segments:
            all_points.extend(segment.points)
        point_aliases = self._canonicalize_points(all_points, tolerance=point_tolerance)
        canonical_segments = [
            CreaseSegment(
                edge_type=segment.edge_type,
                vertex_ids=segment.vertex_ids,
                points=self._ordered_segment_points(
                    self._resolve_point_alias(segment.points[0], point_aliases),
                    self._resolve_point_alias(segment.points[1], point_aliases),
                ),
            )
            for segment in self.segments
        ]

        geometry_errors = []
        boundary_errors = self._validate_boundary_cycle(
            [segment for segment in canonical_segments if segment.edge_type == "B"],
        )
        geometry_errors.extend(boundary_errors)

        for segment in canonical_segments:
            p1, p2 = segment.points
            if self._points_close(p1, p2, tolerance=point_tolerance):
                geometry_errors.append(
                    f"Zero-length segment detected at {self._format_point(p1)} [{segment.edge_type}]."
                )

        planarity_errors = []
        for index, first in enumerate(canonical_segments):
            for second in canonical_segments[index + 1:]:
                issue = self._segment_pair_issue(first, second, point_tolerance=point_tolerance)
                if issue is not None:
                    planarity_errors.append(issue)

        geometry_errors.extend(planarity_errors)

        vertex_lookup = self._vertex_lookup(self.vertices, point_aliases=point_aliases)
        incidence = self._build_incidence(point_aliases=point_aliases)
        vertex_checks = []
        warnings = []
        has_complete_mv_assignment = True

        for point in sorted(incidence):
            incident_edges = incidence[point]
            assignments = tuple(segment.edge_type for segment, _ in incident_edges)
            is_boundary = any(edge_type == "B" for edge_type in assignments)
            notes = []
            issues = []
            sector_angles = ()
            kawasaki_residual = None
            maekawa_difference = None

            if is_boundary:
                notes.append("Boundary vertex: interior flat-foldability theorems were not applied.")
            else:
                crease_incident_edges = [
                    (segment, neighbor)
                    for segment, neighbor in incident_edges
                    if segment.edge_type not in {"B", "F"}
                ]
                degree = len(crease_incident_edges)
                ignored_flat_edges = degree != len(incident_edges)
                directions = []
                for segment, neighbor in crease_incident_edges:
                    start, end = segment.points
                    canonical_start = self._resolve_point_alias(start, point_aliases)
                    canonical_end = self._resolve_point_alias(end, point_aliases)
                    if self._points_close(canonical_start, point, tolerance=point_tolerance):
                        origin = start
                        target = end
                    elif self._points_close(canonical_end, point, tolerance=point_tolerance):
                        origin = end
                        target = start
                    else:
                        origin = point
                        target = neighbor

                    dx = float(target[0]) - float(origin[0])
                    dy = float(target[1]) - float(origin[1])
                    directions.append(math.atan2(dy, dx) % (2.0 * math.pi))

                directions.sort()
                if directions:
                    sector_angles = tuple(
                        (directions[(i + 1) % len(directions)] - directions[i]) % (2.0 * math.pi)
                        for i in range(len(directions))
                    )

                if ignored_flat_edges:
                    notes.append("Ignored flat face-splitting edges when evaluating local flat-foldability.")

                if degree % 2 != 0:
                    issues.append(
                        f"Interior vertex has odd degree {degree}; flat-foldable single vertices must have even degree."
                    )

                if any(angle <= angle_tolerance for angle in sector_angles):
                    issues.append("Interior vertex contains coincident or nearly coincident crease rays.")

                if sector_angles and degree % 2 == 0:
                    kawasaki_residual = abs(sum(sector_angles[::2]) - math.pi)
                    kawasaki_tolerance = max(angle_tolerance, 3e-2) if degree == 2 else angle_tolerance
                    if kawasaki_residual > kawasaki_tolerance:
                        issues.append(
                            "Interior vertex violates Kawasaki's theorem "
                            f"(alternating angle sum error {kawasaki_residual:.6f} radians)."
                        )

                crease_assignments = tuple(
                    segment.edge_type for segment, _ in crease_incident_edges
                )
                if ignored_flat_edges and crease_assignments:
                    has_complete_mv_assignment = False
                    notes.append(
                        "Skipped Maekawa's theorem because the vertex includes ignored flat split edges."
                    )
                    if require_mv_assignment:
                        issues.append("Interior vertex is missing a complete mountain/valley assignment.")
                elif crease_assignments and all(edge_type in {"M", "V"} for edge_type in crease_assignments):
                    maekawa_difference = abs(
                        crease_assignments.count("M") - crease_assignments.count("V")
                    )
                    if maekawa_difference != 2:
                        issues.append(
                            "Interior vertex violates Maekawa's theorem "
                            f"(|M - V| = {maekawa_difference}, expected 2)."
                        )
                elif crease_assignments:
                    has_complete_mv_assignment = False
                    non_mv_types = sorted(
                        {edge_type for edge_type in crease_assignments if edge_type not in {"M", "V"}}
                    )
                    notes.append(
                        "Skipped Maekawa's theorem because the vertex includes non-assigned creases: "
                        + ", ".join(non_mv_types)
                        + "."
                    )
                    if require_mv_assignment:
                        issues.append("Interior vertex is missing a complete mountain/valley assignment.")

            vertex_checks.append(
                VertexFoldabilityCheck(
                    vertex_id=vertex_lookup.get(point),
                    point=point,
                    degree=len(incident_edges),
                    is_boundary=is_boundary,
                    sector_angles=sector_angles,
                    assignments=assignments,
                    kawasaki_residual=kawasaki_residual,
                    maekawa_difference=maekawa_difference,
                    issues=tuple(issues),
                    notes=tuple(notes),
                )
            )

        if not has_complete_mv_assignment:
            warnings.append(
                "Maekawa's theorem was skipped at one or more interior vertices because not every crease is assigned M or V."
            )

        local_errors = [
            issue
            for check in vertex_checks
            if not check.is_boundary
            for issue in check.issues
        ]
        errors = tuple(geometry_errors + local_errors)
        return CreasePatternValidationResult(
            is_valid=not errors,
            is_planar=not planarity_errors,
            satisfies_local_flat_foldability=not local_errors,
            has_complete_mv_assignment=has_complete_mv_assignment,
            errors=errors,
            warnings=tuple(warnings),
            vertex_checks=tuple(vertex_checks),
        )

    def _get_scene_bounds(self) -> tuple[float, float, float, float] | None:
        if not self.vertices:
            return None

        coords = np.array(list(self.vertices.values()))
        return (
            float(np.min(coords[:, 0])),
            float(np.max(coords[:, 0])),
            float(np.min(coords[:, 1])),
            float(np.max(coords[:, 1])),
        )

    def _resolve_square_center(self, square_center: tuple[float, float] | None) -> tuple[float, float]:
        if square_center is not None:
            return float(square_center[0]), float(square_center[1])

        bounds = self._get_scene_bounds()
        if bounds is None:
            return 0.5, 0.5

        min_x, max_x, min_y, max_y = bounds
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def _get_ranges(self) -> tuple[list[float], list[float]]:
        if self._get_scene_bounds() is None:
            return [0.0, 1.0], [0.0, 1.0]

        center_x, center_y = self._resolve_square_center(None)
        half_view = 0.5
        return [center_x - half_view, center_x + half_view], [center_y - half_view, center_y + half_view]

    def _pad_ranges_for_stroke_width(
        self,
        *,
        x_limits: list[float],
        y_limits: list[float],
        width: int,
        edge_styles: dict,
    ) -> tuple[list[float], list[float]]:
        """Expand ranges so boundary strokes aren't clipped at the plot edge.

        Boundary segments sit exactly on the data bounding box, and strokes are
        centered on the path, so without this the outer half of the border's
        width gets clipped by the plot area and it renders half as thick.
        """
        border_width = max(style["width"] for style in edge_styles.values())
        inner_width = max(1, width - PLOT_MARGINS["l"] - PLOT_MARGINS["r"])
        span_x = max(x_limits[1] - x_limits[0], EPS)
        pixels_per_unit = inner_width / span_x
        pad = (border_width / 2.0 + 1.0) / pixels_per_unit
        return (
            [x_limits[0] - pad, x_limits[1] + pad],
            [y_limits[0] - pad, y_limits[1] + pad],
        )

    def plot(self, show: bool = True, save_path: str | None = None) -> None:
        width, height = PLOT_WIDTH_PX, PLOT_HEIGHT_PX
        x_limits, y_limits = self._get_ranges()
        x_limits, y_limits = self._pad_ranges_for_stroke_width(
            x_limits=x_limits,
            y_limits=y_limits,
            width=width,
            edge_styles=EDGE_STYLES,
        )

        if not show:
            if save_path is not None:
                self._save_png_fast(
                    output_path=Path(save_path),
                    width=width,
                    height=height,
                    x_limits=x_limits,
                    y_limits=y_limits,
                )
            return None

        fig = go.Figure()
        for segment in self.segments:
            style = EDGE_STYLES[segment.edge_type]
            fig.add_trace(
                go.Scatter(
                    x=[segment.points[0][0], segment.points[1][0]],
                    y=[segment.points[0][1], segment.points[1][1]],
                    mode="lines",
                    line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                scaleanchor="y",
                scaleratio=1,
                range=x_limits,
            ),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, range=y_limits),
            margin=PLOT_MARGINS,
            width=width,
            height=height,
        )

        if save_path is not None:
            self._save_png_fast(
                output_path=Path(save_path),
                width=width,
                height=height,
                x_limits=x_limits,
                y_limits=y_limits,
            )

        if show:
            resolved_renderer = resolve_plotly_renderer()
            if resolved_renderer is None:
                fig.show()
            else:
                fig.show(renderer=resolved_renderer)

        return None

    def _save_png_fast(
        self,
        *,
        output_path: Path,
        width: int,
        height: int,
        x_limits: list[float],
        y_limits: list[float],
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image, draw, export_scale, scaled_margins = create_canvas(
            width=width,
            height=height,
            image_scale=IMAGE_SCALE,
            background_color="rgba(0,0,0,0)",
            margins=PLOT_MARGINS,
        )
        viewport = RasterViewport(
            x_limits=(float(x_limits[0]), float(x_limits[1])),
            y_limits=(float(y_limits[0]), float(y_limits[1])),
            width=image.width,
            height=image.height,
            margins=scaled_margins,
        )

        for segment in self.segments:
            style = EDGE_STYLES[segment.edge_type]
            draw_line(
                draw,
                viewport,
                segment.points[0],
                segment.points[1],
                color=style["color"],
                width=max(1, style["width"] * export_scale),
                dash=style["dash"],
            )

        image.save(output_path)
