from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
from PIL import ImageDraw

from .config import EPS
from .plotly_support import resolve_plotly_renderer
from .raster import (
    RasterViewport,
    create_canvas,
    draw_centered_text,
    draw_centered_text_box,
    draw_line,
    draw_polygon,
    rgba,
)

if TYPE_CHECKING:
    from .core import OrigamiCore

PointT = tuple[float, float]
SegmentT = tuple[PointT, PointT]


BG_COLOR = 'lightgray'
EDGE_COLOR = 'black'
FLAT_EDGE_DASH_PATTERN = '5px,7px'
VERTEX_LABEL_COLOR = 'black'
VERTEX_BORDER_COLOR = 'black'
FACE_LABEL_COLOR = 'black'
PLOT_MARGINS = dict(l=20, r=20, t=40, b=20)
FACE_LABEL_FONT_SIZE = 12
FACE_LABEL_MIN_PIXEL_DISTANCE = 14
PLOT_SIZE_PX = 560  # fixed "medium" figure size
VERTEX_SIZE_SCALE = 0.8
IMAGE_SCALE = 2.0
# Imported representations are commonly rounded to 6 decimals, so keep the
# visual snapping tolerance comfortably above that noise floor without
# approaching the much larger modeling epsilon.
VISUAL_EPS = min(EPS, 1e-5)

class OrigamiVisualizer:
    def __init__(self, origami: OrigamiCore) -> None:
        self.origami = origami

    def _get_face_color(self, orientation: int) -> str:
        if orientation == 0:
            return self.origami.paper_front_color
        return self.origami.paper_back_color

    def _collect_scene_data(
        self,
        draw_order: list[int],
    ) -> tuple[list[dict], dict[SegmentT, dict], dict[int, int], list[list[list[PointT]]]]:
        """Build face/edge data once so Plotly and static export stay aligned."""
        face_records = []
        segment_records = []
        plane_height = {pid: idx for idx, pid in enumerate(draw_order)}
        polygons_by_height = [[] for _ in draw_order]

        for pid in draw_order:
            plane = self.origami.planes[pid]
            for fid in plane.face_ids:
                face = self.origami.faces[fid]
                edges = self.origami._walk_face_half_edges(face)

                coords = [he.origin.pos for he in edges]
                coords.append(edges[0].origin.pos)
                coords = np.array(coords)
                polygon = [tuple(p) for p in coords[:-1]]
                polygons_by_height[plane_height[pid]].append(polygon)
                face_records.append(
                    {
                        'fid': fid,
                        'pid': pid,
                        'orientation': face.orientation,
                        'coords': coords,
                        'polygon': polygon,
                    }
                )

                for he in edges:
                    v1, v2 = he.origin.pos, he.next.origin.pos
                    twin_type = he.twin.type if he.twin is not None else None
                    style = 'dash' if (he.type == 'F' or twin_type == 'F') else 'solid'
                    segment_records.append(
                        {
                            'p1': tuple(v1),
                            'p2': tuple(v2),
                            'pid': pid,
                            'style': style,
                        }
                    )

        segment_data = self._build_segment_data(segment_records)
        return face_records, segment_data, plane_height, polygons_by_height

    def _get_scene_bounds(self, face_records: list[dict]) -> tuple[float, float, float, float] | None:
        all_points = []
        for record in face_records:
            all_points.extend(record['polygon'])

        if not all_points:
            return None

        coords = np.array(all_points)
        return (
            float(np.min(coords[:, 0])),
            float(np.max(coords[:, 0])),
            float(np.min(coords[:, 1])),
            float(np.max(coords[:, 1])),
        )

    def _resolve_square_center(self, face_records: list[dict], square_center: PointT | None) -> tuple[float, float]:
        if square_center is not None:
            return float(square_center[0]), float(square_center[1])

        bounds = self._get_scene_bounds(face_records)
        if bounds is None:
            return 0.5, 0.5

        min_x, max_x, min_y, max_y = bounds
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def _get_fixed_square_frame(
        self,
        face_records: list[dict],
        square_center: PointT | None,
        square_size: float,
        square_padding_ratio: float,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        center_x, center_y = self._resolve_square_center(face_records, square_center)
        square_side = max(float(square_size), EPS)
        half_square = 0.5 * square_side
        half_view = half_square * (1 + max(square_padding_ratio, 0.0))
        square_x = [center_x - half_square, center_x + half_square]
        square_y = [center_y - half_square, center_y + half_square]
        x_range = [center_x - half_view, center_x + half_view]
        y_range = [center_y - half_view, center_y + half_view]
        return x_range, y_range, square_x, square_y

    def _get_draw_order(self) -> list[int]:
        """Returns plane IDs from bottom to top based on the layering graph."""
        if not nx.is_directed_acyclic_graph(self.origami.layers):
            return list(self.origami.planes.keys())

        remaining = self.origami.layers.copy()
        draw_order = []

        while remaining.nodes:
            bottom_layer = [pid for pid in remaining.nodes if remaining.out_degree(pid) == 0]
            draw_order.extend(sorted(bottom_layer))
            remaining.remove_nodes_from(bottom_layer)

        return draw_order

    def _segment_key(self, v1: PointT, v2: PointT) -> SegmentT:
        """Returns a stable undirected key for a segment, snapped to visualizer tolerance."""
        p1 = (
            round(v1[0] / VISUAL_EPS) * VISUAL_EPS,
            round(v1[1] / VISUAL_EPS) * VISUAL_EPS,
        )
        p2 = (
            round(v2[0] / VISUAL_EPS) * VISUAL_EPS,
            round(v2[1] / VISUAL_EPS) * VISUAL_EPS,
        )
        return (p1, p2) if p1 <= p2 else (p2, p1)

    def _project_point_to_segment(self, point: PointT, start: PointT, end: PointT) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_squared = dx * dx + dy * dy
        if length_squared <= VISUAL_EPS:
            return 0.0
        return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared

    def _build_segment_data(self, segment_records: list[dict]) -> dict[SegmentT, dict]:
        if not segment_records:
            return {}

        all_points = [record['p1'] for record in segment_records]
        all_points.extend(record['p2'] for record in segment_records)

        segment_data = {}
        for record in segment_records:
            p1 = record['p1']
            p2 = record['p2']
            split_params = [0.0, 1.0]

            for point in all_points:
                if not self._point_on_segment(point, p1, p2):
                    continue
                param = self._project_point_to_segment(point, p1, p2)
                if VISUAL_EPS < param < 1.0 - VISUAL_EPS:
                    split_params.append(min(1.0, max(0.0, param)))

            split_params = sorted(split_params)
            deduped_params = []
            for param in split_params:
                if not deduped_params or abs(param - deduped_params[-1]) > VISUAL_EPS:
                    deduped_params.append(param)

            for start_param, end_param in zip(deduped_params, deduped_params[1:]):
                if end_param - start_param <= VISUAL_EPS:
                    continue

                start = (
                    p1[0] + start_param * (p2[0] - p1[0]),
                    p1[1] + start_param * (p2[1] - p1[1]),
                )
                end = (
                    p1[0] + end_param * (p2[0] - p1[0]),
                    p1[1] + end_param * (p2[1] - p1[1]),
                )
                key = self._segment_key(start, end)
                if key not in segment_data:
                    segment_data[key] = {
                        'owner_pids': set(),
                        'styles_by_pid': defaultdict(set),
                    }
                segment_data[key]['owner_pids'].add(record['pid'])
                segment_data[key]['styles_by_pid'][record['pid']].add(record['style'])

        return segment_data

    def _point_on_segment(self, p: PointT, a: PointT, b: PointT) -> bool:
        """Checks if point p lies on segment ab within visualizer tolerance."""
        ax, ay = a
        bx, by = b
        px, py = p
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        cross = abx * apy - aby * apx
        if abs(cross) > VISUAL_EPS:
            return False
        dot = apx * abx + apy * aby
        if dot < -VISUAL_EPS:
            return False
        ab2 = abx * abx + aby * aby
        if dot - ab2 > VISUAL_EPS:
            return False
        return True

    def _point_in_polygon_status(self, point: PointT, polygon: list[PointT]) -> str:
        """Returns 'inside', 'boundary', or 'outside' for a 2D point and polygon."""
        x, y = point
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            if self._point_on_segment((x, y), (x1, y1), (x2, y2)):
                return 'boundary'

            intersects = ((y1 > y) != (y2 > y))
            if intersects:
                x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x_cross > x:
                    inside = not inside

        return 'inside' if inside else 'outside'

    def _cross_2d(self, a: PointT, b: PointT) -> float:
        return a[0] * b[1] - a[1] * b[0]

    def _points_close(self, p1: PointT, p2: PointT) -> bool:
        return abs(p1[0] - p2[0]) <= VISUAL_EPS and abs(p1[1] - p2[1]) <= VISUAL_EPS

    def _segment_direction(self, segment: SegmentT) -> PointT:
        (x1, y1), (x2, y2) = segment
        return (x2 - x1, y2 - y1)

    def _segment_length(self, segment: SegmentT) -> float:
        dx, dy = self._segment_direction(segment)
        return float(np.hypot(dx, dy))

    def _are_collinear_segments(self, seg1: SegmentT, seg2: SegmentT) -> bool:
        dir1 = self._segment_direction(seg1)
        dir2 = self._segment_direction(seg2)
        len1 = self._segment_length(seg1)
        len2 = self._segment_length(seg2)
        if len1 <= VISUAL_EPS or len2 <= VISUAL_EPS:
            return False

        if abs(self._cross_2d(dir1, dir2)) > VISUAL_EPS * len1 * len2:
            return False

        base = (seg2[0][0] - seg1[0][0], seg2[0][1] - seg1[0][1])
        distance_to_supporting_line = abs(self._cross_2d(dir1, base)) / len1
        return distance_to_supporting_line <= VISUAL_EPS

    def _merge_collinear_segments(self, seg1: SegmentT, seg2: SegmentT) -> SegmentT | None:
        if not self._are_collinear_segments(seg1, seg2):
            return None

        direction = self._segment_direction(seg1)
        alt_direction = self._segment_direction(seg2)
        if abs(alt_direction[0]) + abs(alt_direction[1]) > abs(direction[0]) + abs(direction[1]):
            direction = alt_direction

        if abs(direction[0]) + abs(direction[1]) <= VISUAL_EPS:
            return seg2 if self._points_close(seg1[0], seg1[1]) else seg1

        axis = 0 if abs(direction[0]) >= abs(direction[1]) else 1
        epsilon = VISUAL_EPS if axis == 0 else VISUAL_EPS
        sign = 1.0 if direction[axis] >= 0 else -1.0

        def _coord(point: PointT) -> float:
            return sign * point[axis]

        projections = [
            (_coord(seg1[0]), seg1[0]),
            (_coord(seg1[1]), seg1[1]),
            (_coord(seg2[0]), seg2[0]),
            (_coord(seg2[1]), seg2[1]),
        ]
        seg1_lo, seg1_hi = sorted((_coord(seg1[0]), _coord(seg1[1])))
        seg2_lo, seg2_hi = sorted((_coord(seg2[0]), _coord(seg2[1])))

        if min(seg1_hi, seg2_hi) < max(seg1_lo, seg2_lo) - epsilon:
            return None

        start = min(projections, key=lambda item: item[0])[1]
        end = max(projections, key=lambda item: item[0])[1]
        return start, end

    def _merge_drawn_segments(self, segments: list[dict]) -> list[dict]:
        if not segments:
            return []

        merged_segments = []
        by_style = defaultdict(list)
        for segment in segments:
            by_style[segment['style']].append(segment)

        for style, style_segments in by_style.items():
            remaining = style_segments[:]

            while remaining:
                current = remaining.pop()
                start, end = current['p1'], current['p2']

                extended = True
                while extended:
                    extended = False

                    for index, candidate in enumerate(remaining):
                        candidate_segment = (candidate['p1'], candidate['p2'])
                        current_segment = (start, end)
                        merged = self._merge_collinear_segments(current_segment, candidate_segment)
                        if merged is None:
                            continue

                        start, end = merged

                        remaining.pop(index)
                        extended = True
                        break

                if self._points_close(start, end):
                    continue

                merged_segments.append({'p1': start, 'p2': end, 'style': style})

        return merged_segments

    def _segment_intersection_params(
        self,
        p1: PointT,
        p2: PointT,
        q1: PointT,
        q2: PointT,
    ) -> list[float | tuple[float, float]]:
        """Return parameter values on p1-p2 where it intersects q1-q2."""
        r = (p2[0] - p1[0], p2[1] - p1[1])
        s = (q2[0] - q1[0], q2[1] - q1[1])
        qmp = (q1[0] - p1[0], q1[1] - p1[1])
        rxs = self._cross_2d(r, s)
        qmpxr = self._cross_2d(qmp, r)

        if abs(rxs) <= VISUAL_EPS:
            if abs(qmpxr) > VISUAL_EPS:
                return []

            rr = r[0] * r[0] + r[1] * r[1]
            if rr <= VISUAL_EPS:
                return []

            t0 = ((q1[0] - p1[0]) * r[0] + (q1[1] - p1[1]) * r[1]) / rr
            t1 = ((q2[0] - p1[0]) * r[0] + (q2[1] - p1[1]) * r[1]) / rr
            lo = max(0.0, min(t0, t1))
            hi = min(1.0, max(t0, t1))
            if hi < -VISUAL_EPS or lo > 1.0 + VISUAL_EPS or hi - lo <= VISUAL_EPS:
                return []
            return [(max(0.0, lo), min(1.0, hi))]

        t = self._cross_2d(qmp, s) / rxs
        u = self._cross_2d(qmp, r) / rxs
        if -VISUAL_EPS <= t <= 1.0 + VISUAL_EPS and -VISUAL_EPS <= u <= 1.0 + VISUAL_EPS:
            return [min(1.0, max(0.0, t))]
        return []

    def _merge_intervals(self, intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not intervals:
            return []

        normalized = []
        for start, end in intervals:
            lo = max(0.0, min(float(start), float(end)))
            hi = min(1.0, max(float(start), float(end)))
            if hi - lo > VISUAL_EPS:
                normalized.append((lo, hi))

        if not normalized:
            return []

        normalized.sort()
        merged = [normalized[0]]
        for start, end in normalized[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + VISUAL_EPS:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    def _subtract_intervals(
        self,
        base_intervals: list[tuple[float, float]],
        cut_intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if not cut_intervals:
            return list(base_intervals)

        remaining = []
        for base_start, base_end in base_intervals:
            cursor = base_start
            for cut_start, cut_end in cut_intervals:
                if cut_end <= cursor + VISUAL_EPS:
                    continue
                if cut_start >= base_end - VISUAL_EPS:
                    break
                if cut_start > cursor + VISUAL_EPS:
                    remaining.append((cursor, min(base_end, cut_start)))
                cursor = max(cursor, cut_end)
                if cursor >= base_end - VISUAL_EPS:
                    break
            if cursor < base_end - VISUAL_EPS:
                remaining.append((cursor, base_end))
        return remaining

    def _segment_cover_intervals_by_polygon(
        self,
        p1: PointT,
        p2: PointT,
        polygon: list[PointT],
    ) -> list[tuple[float, float]]:
        ts = [0.0, 1.0]
        n = len(polygon)

        for i in range(n):
            q1 = polygon[i]
            q2 = polygon[(i + 1) % n]
            for hit in self._segment_intersection_params(p1, p2, q1, q2):
                if isinstance(hit, tuple):
                    ts.extend(hit)
                else:
                    ts.append(hit)

        ts = sorted(min(1.0, max(0.0, t)) for t in ts)
        deduped = []
        for t in ts:
            if not deduped or abs(t - deduped[-1]) > VISUAL_EPS:
                deduped.append(t)

        covered = []
        for start, end in zip(deduped, deduped[1:]):
            if end - start <= VISUAL_EPS:
                continue
            mid = 0.5 * (start + end)
            point = (
                p1[0] + mid * (p2[0] - p1[0]),
                p1[1] + mid * (p2[1] - p1[1]),
            )
            # A polygon that only touches an edge on its own boundary should not
            # erase that outline from the final drawing.
            if self._point_in_polygon_status(point, polygon) == 'inside':
                covered.append((start, end))

        return self._merge_intervals(covered)

    def _get_visible_segment_parts(
        self,
        p1: PointT,
        p2: PointT,
        owner_height: int,
        polygons_by_height: list[list[list[PointT]]],
    ) -> list[SegmentT]:
        covered_intervals = []
        for height in range(owner_height + 1, len(polygons_by_height)):
            for poly in polygons_by_height[height]:
                covered_intervals.extend(
                    self._segment_cover_intervals_by_polygon(p1, p2, poly)
                )

        visible_intervals = self._subtract_intervals(
            [(0.0, 1.0)],
            self._merge_intervals(covered_intervals),
        )

        visible_parts = []
        for start, end in visible_intervals:
            if end - start <= VISUAL_EPS:
                continue
            visible_parts.append((
                (
                    p1[0] + start * (p2[0] - p1[0]),
                    p1[1] + start * (p2[1] - p1[1]),
                ),
                (
                    p1[0] + end * (p2[0] - p1[0]),
                    p1[1] + end * (p2[1] - p1[1]),
                ),
            ))

        return visible_parts

    def _is_segment_occluded(
        self,
        p1: PointT,
        p2: PointT,
        owner_height: int,
        polygons_by_height: list[list[list[PointT]]],
    ) -> bool:
        """True if segment p1-p2 is fully covered by strictly higher layers."""
        return len(self._get_visible_segment_parts(p1, p2, owner_height, polygons_by_height)) == 0

    def _get_plot_pixel_dimensions(self, width: float, height: float) -> tuple[float, float]:
        plot_width = max(float(width) - PLOT_MARGINS["l"] - PLOT_MARGINS["r"], 1.0)
        plot_height = max(float(height) - PLOT_MARGINS["t"] - PLOT_MARGINS["b"], 1.0)
        return plot_width, plot_height

    def _polygon_area(self, polygon: list[PointT]) -> float:
        if len(polygon) < 3:
            return 0.0

        area = 0.0
        for index, point in enumerate(polygon):
            next_point = polygon[(index + 1) % len(polygon)]
            area += point[0] * next_point[1] - next_point[0] * point[1]
        return abs(area) * 0.5

    def _point_to_pixels(
        self,
        point: PointT,
        x_limits: list[float],
        y_limits: list[float],
        plot_width: float,
        plot_height: float,
    ) -> tuple[float, float]:
        x_span = max(x_limits[1] - x_limits[0], VISUAL_EPS)
        y_span = max(y_limits[1] - y_limits[0], VISUAL_EPS)
        px = (point[0] - x_limits[0]) / x_span * plot_width
        py = (point[1] - y_limits[0]) / y_span * plot_height
        return px, py

    def _get_plotly_dash_style(self, style: str) -> str:
        return FLAT_EDGE_DASH_PATTERN if style == "dash" else "solid"

    def _filter_face_label_candidates(
        self,
        label_candidates: list[dict],
        x_limits: list[float],
        y_limits: list[float],
        plot_width: float,
        plot_height: float,
    ) -> list[dict]:
        if not label_candidates:
            return []

        kept = []
        kept_pixels = []
        ordered_candidates = sorted(
            label_candidates,
            key=lambda candidate: (
                -candidate["area"],
                -candidate["height"],
                candidate["fid"],
            ),
        )

        for candidate in ordered_candidates:
            pixel_point = self._point_to_pixels(
                candidate["centroid"],
                x_limits=x_limits,
                y_limits=y_limits,
                plot_width=plot_width,
                plot_height=plot_height,
            )
            overlaps_existing = any(
                (pixel_point[0] - existing[0]) ** 2 + (pixel_point[1] - existing[1]) ** 2
                < FACE_LABEL_MIN_PIXEL_DISTANCE ** 2
                for existing in kept_pixels
            )
            if overlaps_existing:
                continue

            kept.append(candidate)
            kept_pixels.append(pixel_point)

        return sorted(kept, key=lambda candidate: candidate["fid"])

    def _is_point_occluded(
        self,
        point: PointT,
        owner_height: int,
        polygons_by_height: list[list[list[PointT]]],
    ) -> bool:
        for height in range(owner_height + 1, len(polygons_by_height)):
            for polygon in polygons_by_height[height]:
                if self._point_in_polygon_status(point, polygon) in {"inside", "boundary"}:
                    return True
        return False

    def _get_face_label_point(
        self,
        polygon: list[PointT],
        owner_height: int,
        polygons_by_height: list[list[list[PointT]]],
    ) -> PointT | None:
        if not polygon:
            return None

        coords = np.array(polygon, dtype=float)
        centroid = np.mean(coords, axis=0)

        candidates = [tuple(centroid)]
        for vertex in coords:
            candidates.append(tuple(0.7 * centroid + 0.3 * vertex))

        for index, vertex in enumerate(coords):
            next_vertex = coords[(index + 1) % len(coords)]
            edge_midpoint = 0.5 * (vertex + next_vertex)
            candidates.append(tuple(0.65 * centroid + 0.35 * edge_midpoint))

        deduped_candidates = []
        for candidate in candidates:
            if any(self._points_close(candidate, existing) for existing in deduped_candidates):
                continue
            deduped_candidates.append(candidate)

        for candidate in deduped_candidates:
            if self._point_in_polygon_status(candidate, polygon) == "outside":
                continue
            if self._is_point_occluded(candidate, owner_height, polygons_by_height):
                continue
            return candidate

        return None

    def plot(
        self,
        show_vertices_indices: bool = False,
        show_faces_indices: bool = False,
        show: bool = True,
        save_path: str | None = None,
    ) -> None:
        width = height = PLOT_SIZE_PX
        draw_order = self._get_draw_order()

        face_records, segment_data, plane_height, polygons_by_height = self._collect_scene_data(draw_order)
        x_limits, y_limits, _, _ = self._get_fixed_square_frame(
            face_records=face_records,
            square_center=None,
            square_size=1.0,
            square_padding_ratio=1,
        )
        plot_width, plot_height = self._get_plot_pixel_dimensions(width, height)

        if not show:
            if save_path is not None:
                self._save_png_fast(
                    output_path=Path(save_path),
                    draw_order=draw_order,
                    face_records=face_records,
                    segment_data=segment_data,
                    plane_height=plane_height,
                    polygons_by_height=polygons_by_height,
                    x_limits=x_limits,
                    y_limits=y_limits,
                    width=width,
                    height=height,
                    show_vertices_indices=show_vertices_indices,
                    show_faces_indices=show_faces_indices,
                    plot_width=plot_width,
                    plot_height=plot_height,
                )
            return None

        fig = go.Figure()

        # 1. Draw faces (bottom-to-top) and collect face-label candidates.
        face_label_candidates = []
        for record in face_records:
            coords = record['coords']
            face_color = self._get_face_color(record["orientation"])

            fig.add_trace(go.Scatter(
                x=coords[:, 0], y=coords[:, 1],
                fill="toself",
                fillcolor=face_color,
                line=dict(color='rgba(0,0,0,0)'),
                mode='lines',
                showlegend=False,
                hoverinfo='skip',
            ))

            if show_faces_indices:
                label_point = self._get_face_label_point(
                    record['polygon'],
                    owner_height=plane_height[record['pid']],
                    polygons_by_height=polygons_by_height,
                )
                if label_point is None:
                    continue
                face_label_candidates.append(
                    {
                        'fid': record['fid'],
                        'centroid': (float(label_point[0]), float(label_point[1])),
                        'area': self._polygon_area(record['polygon']),
                        'height': plane_height[record['pid']],
                    }
                )

        if show_faces_indices:
            for candidate in self._filter_face_label_candidates(
                face_label_candidates,
                x_limits=x_limits,
                y_limits=y_limits,
                plot_width=plot_width,
                plot_height=plot_height,
            ):
                fig.add_trace(go.Scatter(
                    x=[candidate['centroid'][0]], y=[candidate['centroid'][1]],
                    mode='text',
                    text=[str(candidate['fid'])],
                    textfont=dict(color=FACE_LABEL_COLOR, size=FACE_LABEL_FONT_SIZE, family="Arial"),
                    showlegend=False,
                    hoverinfo='none'
                ))

        # 2. Collect visible edge segments, then merge collinear runs to avoid
        # rasterization artifacts from many tiny adjacent traces.
        drawable_segments = []
        for (p1, p2), info in segment_data.items():
            owners = info['owner_pids']
            top_pid = max(owners, key=lambda p: plane_height[p])
            owner_height = plane_height[top_pid]
            owner_styles = info['styles_by_pid'][top_pid]
            line_style = 'dash' if 'dash' in owner_styles else 'solid'

            for visible_p1, visible_p2 in self._get_visible_segment_parts(p1, p2, owner_height, polygons_by_height):
                drawable_segments.append(
                    {'p1': visible_p1, 'p2': visible_p2, 'style': line_style}
                )

        for segment in self._merge_drawn_segments(drawable_segments):
            fig.add_trace(go.Scatter(
                x=[segment['p1'][0], segment['p2'][0]],
                y=[segment['p1'][1], segment['p2'][1]],
                mode='lines',
                line=dict(
                    color='black',
                    width=2,
                    shape='linear',
                    dash=self._get_plotly_dash_style(segment['style']),
                ),
                hoverinfo='none',
                showlegend=False
            ))

        # 3. Add stacked vertex labels.
        if show_vertices_indices:
            self._add_vertex_traces(fig, draw_order)

        # 4. Layout styling.
        fig.update_layout(
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=BG_COLOR,
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                scaleanchor="y",
                scaleratio=1,
                range=x_limits,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                range=y_limits,
            ),
            margin=PLOT_MARGINS,
            width=width,
            height=height,
        )

        if save_path is not None:
            self._save_png_fast(
                output_path=Path(save_path),
                draw_order=draw_order,
                face_records=face_records,
                segment_data=segment_data,
                plane_height=plane_height,
                polygons_by_height=polygons_by_height,
                x_limits=x_limits,
                y_limits=y_limits,
                width=width,
                height=height,
                show_vertices_indices=show_vertices_indices,
                show_faces_indices=show_faces_indices,
                plot_width=plot_width,
                plot_height=plot_height,
            )

        resolved_renderer = resolve_plotly_renderer()
        if resolved_renderer is None:
            fig.show()
        else:
            fig.show(renderer=resolved_renderer)

        return None

    def _collect_face_label_candidates(
        self,
        face_records: list[dict],
        plane_height: dict[int, int],
        polygons_by_height: list[list[list[PointT]]],
    ) -> list[dict]:
        candidates = []
        for record in face_records:
            label_point = self._get_face_label_point(
                record["polygon"],
                owner_height=plane_height[record["pid"]],
                polygons_by_height=polygons_by_height,
            )
            if label_point is None:
                continue
            candidates.append(
                {
                    "fid": record["fid"],
                    "centroid": (float(label_point[0]), float(label_point[1])),
                    "area": self._polygon_area(record["polygon"]),
                    "height": plane_height[record["pid"]],
                }
            )
        return candidates

    def _collect_drawable_segments(
        self,
        segment_data: dict[SegmentT, dict],
        plane_height: dict[int, int],
        polygons_by_height: list[list[list[PointT]]],
    ) -> list[dict]:
        drawable_segments = []
        for (p1, p2), info in segment_data.items():
            owners = info["owner_pids"]
            top_pid = max(owners, key=lambda pid: plane_height[pid])
            owner_height = plane_height[top_pid]
            owner_styles = info["styles_by_pid"][top_pid]
            line_style = "dash" if "dash" in owner_styles else "solid"
            visible_parts = self._get_visible_segment_parts(p1, p2, owner_height, polygons_by_height)

            for visible_p1, visible_p2 in visible_parts:
                drawable_segments.append(
                    {"p1": visible_p1, "p2": visible_p2, "style": line_style}
                )
        return self._merge_drawn_segments(drawable_segments)

    def _draw_vertex_labels_png(
        self,
        draw: ImageDraw.ImageDraw,
        viewport: RasterViewport,
        draw_order: list[int],
        *,
        export_scale: float,
    ) -> None:
        grouped_vertices = self._group_vertices_by_position(draw_order=draw_order)
        marker_size, border_width, text_size = self._get_vertex_marker_style()
        block_padding = max(0, int(round(marker_size * 0.08 * export_scale)))

        for x, y, vids in grouped_vertices:
            draw_centered_text_box(
                draw,
                viewport,
                (x, y),
                str(vids[0]) if len(vids) == 1 else self._format_vertex_block_text(vids),
                font_size=max(6, text_size * (0.85 if len(vids) == 1 else 0.72) * export_scale),
                text_color=VERTEX_LABEL_COLOR,
                fill_color=rgba("white", 192),
                outline_color=VERTEX_BORDER_COLOR,
                padding=block_padding,
                spacing=max(2, int(round(2 * export_scale))),
            )

    def _save_png_fast(
        self,
        *,
        output_path: Path,
        draw_order: list[int],
        face_records: list[dict],
        segment_data: dict[SegmentT, dict],
        plane_height: dict[int, int],
        polygons_by_height: list[list[list[PointT]]],
        x_limits: list[float],
        y_limits: list[float],
        width: int,
        height: int,
        show_vertices_indices: bool,
        show_faces_indices: bool,
        plot_width: int,
        plot_height: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image, draw, export_scale, scaled_margins = create_canvas(
            width=width,
            height=height,
            image_scale=IMAGE_SCALE,
            background_color=BG_COLOR,
            margins=PLOT_MARGINS,
        )
        viewport = RasterViewport(
            x_limits=(float(x_limits[0]), float(x_limits[1])),
            y_limits=(float(y_limits[0]), float(y_limits[1])),
            width=image.width,
            height=image.height,
            margins=scaled_margins,
        )

        for record in face_records:
            face_color = self._get_face_color(record["orientation"])
            draw_polygon(draw, viewport, record["polygon"], fill_color=face_color)

        if show_faces_indices:
            for candidate in self._filter_face_label_candidates(
                self._collect_face_label_candidates(face_records, plane_height, polygons_by_height),
                x_limits=x_limits,
                y_limits=y_limits,
                plot_width=plot_width,
                plot_height=plot_height,
            ):
                draw_centered_text(
                    draw,
                    viewport,
                    candidate["centroid"],
                    str(candidate["fid"]),
                    font_size=max(8, FACE_LABEL_FONT_SIZE * export_scale),
                    text_color=FACE_LABEL_COLOR,
                )

        for segment in self._collect_drawable_segments(
            segment_data=segment_data,
            plane_height=plane_height,
            polygons_by_height=polygons_by_height,
        ):
            draw_line(
                draw,
                viewport,
                segment["p1"],
                segment["p2"],
                color=EDGE_COLOR,
                width=max(1, int(round(2 * export_scale))),
                dash=segment["style"],
            )

        if show_vertices_indices:
            self._draw_vertex_labels_png(
                draw,
                viewport,
                draw_order,
                export_scale=export_scale,
            )

        image.save(output_path)

    def _get_vertex_marker_style(self) -> tuple[float, float, int]:
        marker_size = 18 * VERTEX_SIZE_SCALE
        border_width = max(1.0, 0.08 * marker_size)
        # Keep text near the circle boundary without touching it.
        text_size = max(7, int(round(marker_size * 0.80)))
        return marker_size, border_width, text_size

    def _group_vertices_by_position(self, draw_order: list[int]) -> list[tuple[float, float, list[int]]]:
        pos_groups = defaultdict(list)
        for vid, v in self.origami.vertices.items():
            # Snap to grid to handle float precision
            key = (
                round(v.pos[0] / VISUAL_EPS) * VISUAL_EPS,
                round(v.pos[1] / VISUAL_EPS) * VISUAL_EPS,
            )
            pos_groups[key].append(vid)

        if not pos_groups:
            return []

        plane_height = {pid: idx for idx, pid in enumerate(draw_order)}
        vertex_height = {}
        for pid in draw_order:
            height = plane_height[pid]
            for fid in self.origami.planes[pid].face_ids:
                for vid in self.origami._face_vids(fid):
                    vertex_height[vid] = max(vertex_height.get(vid, -1), height)

        def get_max_height(vid: int) -> int:
            # Higher index in draw_order = visually "on top"
            return vertex_height.get(vid, -1)

        grouped_vertices = []
        for pos, vids in pos_groups.items():
            # Sort vids by their height in the stack (topmost first)
            sorted_vids = sorted(vids, key=get_max_height, reverse=True)
            grouped_vertices.append((pos[0], pos[1], sorted_vids))

        return grouped_vertices

    def _format_vertex_block_text(self, vids: list[int]) -> str:
        # Preserve the stack order visually from top to bottom.
        return "<br>".join(str(vid) for vid in vids)

    def _add_vertex_traces(self, fig: go.Figure, draw_order: list[int]) -> None:
        grouped_vertices = self._group_vertices_by_position(draw_order=draw_order)
        marker_size, border_width, text_size = self._get_vertex_marker_style()
        block_padding = max(0, int(round(marker_size * 0.08)))

        for x, y, vids in grouped_vertices:
            fig.add_annotation(
                x=x,
                y=y,
                text=str(vids[0]) if len(vids) == 1 else self._format_vertex_block_text(vids),
                showarrow=False,
                xanchor='center',
                yanchor='middle',
                align='center',
                font=dict(
                    color=VERTEX_LABEL_COLOR,
                    size=max(6, int(round(text_size * (0.85 if len(vids) == 1 else 0.72)))),
                ),
                bgcolor='rgba(255,255,255,0.75)',
                bordercolor=VERTEX_BORDER_COLOR,
                borderwidth=max(1.0, border_width * 0.5),
                borderpad=block_padding,
            )
