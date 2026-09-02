from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import EPS
from .DCEL import Face, HalfEdge
from .utils import (
    Line,
    find_linear_paths_nx,
    get_line_equation,
    point_side_to_line,
    segment_line_intersection,
)


class OrigamiGeometryMixin:
    """DCEL/geometry queries: lines, sides, faces, and edges by position."""

    def _get_line_equation(self, edge: Tuple[int, int]) -> Line:
        v1_id, v2_id = edge
        self._check_vid_exists(v1_id)
        self._check_vid_exists(v2_id)
        p1, p2 = self.vertices[v1_id].pos, self.vertices[v2_id].pos
        line_eq = get_line_equation(p1, p2)
        return line_eq

    def _get_vid_side(self, vid: int, line: Line) -> int:
        side = point_side_to_line(self.vertices[vid].pos, line)
        return side

    def _walk_face_half_edges(self, face: Face) -> list[HalfEdge]:
        """Safely walk a face boundary even if the DCEL is already malformed."""
        if face.edge is None:
            return []

        half_edges = []
        visited_half_edges = set()
        current = face.edge
        while current is not None and id(current) not in visited_half_edges:
            visited_half_edges.add(id(current))
            half_edges.append(current)
            current = current.next
        return half_edges

    def _face_vids(self, fid: int) -> list[int]:
        face = self.faces[fid]
        return [edge.origin.id for edge in self._walk_face_half_edges(face) if edge.origin is not None]

    def _face_has_edge(self, face: Face, edge: Tuple[int, int]) -> bool:
        target = tuple(sorted(edge))
        for half_edge in self._walk_face_half_edges(face):
            if half_edge.origin is None or half_edge.next is None or half_edge.next.origin is None:
                continue
            boundary_edge = tuple(sorted((half_edge.origin.id, half_edge.next.origin.id)))
            if boundary_edge == target:
                return True
        return False

    def _is_there_face_on_side_of_edge(self, edge: Tuple[int, int], side: int) -> bool:
        fid, _ = self._get_edge_faces_by_side(edge, side)
        face = self.faces[fid]
        line = self._get_line_equation(edge)
        for curr_edge in self._walk_face_half_edges(face):
            vid = curr_edge.origin.id
            if vid in edge:
                continue
            vid_side = self._get_vid_side(vid, line)
            if vid_side != side:
                return False
        return True

    def _get_edge_faces_by_side(self, edge: Tuple[int, int], side: int) -> Tuple[int, int | None]:
        self.root_logger.debug(f"Getting faces on side {side} of edge {edge}")
        line = self._get_line_equation(edge)
        he = self.half_edges[edge]
        face = he.face
        twin_face = he.twin.face
        for curr_edge in self._walk_face_half_edges(face):
            vid = curr_edge.origin.id
            if vid in edge:
                continue
            # Find the first vertex that is not part of the given edge
            vid_side = self._get_vid_side(vid, line)
            other_face_id = twin_face.id if twin_face else None
            if vid_side == side:
                # This face is on the desired side of the line
                self.root_logger.debug(f"Face on side {side}: face {face.id}. Other side: {other_face_id}")
                return face.id, other_face_id
            # This face is not on the desired side
            self.root_logger.debug(f"Face on side {side}: {other_face_id}. Other side: {face.id}")
            return other_face_id, face.id

        face_vertices = self._face_vids(face.id)
        raise ValueError(
            f"Could not determine which side of {edge} contains face {face.id}. "
            f"The face boundary is degenerate or lies entirely on the fold line: {face_vertices}"
        )

    def _get_face_containing_the_two_vertices(self, vid1: int, vid2: int) -> int | None:
        for face in self.faces.values():
            edge = face.edge
            vertices_in_face = set()
            while True:

                vertices_in_face.add(edge.origin.id)
                edge = edge.next
                if edge == face.edge:
                    break
            if vid1 in vertices_in_face and vid2 in vertices_in_face:
                return face.id
        return None

    def _side_by_vid(self, line: Line) -> dict[int, int]:
        # Map every vertex id to which side of `line` it's on.
        map = {}
        for vid in self.vertices:
            side = self._get_vid_side(vid, line)
            map[vid] = side
        return map

    def _edges_are_the_same(self, edge1: Tuple[int, int], edge2: Tuple[int, int]) -> bool:
        if np.allclose(self.vertices[edge1[0]].pos, self.vertices[edge2[0]].pos) and np.allclose(self.vertices[edge1[1]].pos, self.vertices[edge2[1]].pos):
            return True
        if np.allclose(self.vertices[edge1[0]].pos, self.vertices[edge2[1]].pos) and np.allclose(self.vertices[edge1[1]].pos, self.vertices[edge2[0]].pos):
            return True
        return False

    def _get_edge_line_intersection(self, line: Line, edge: Tuple[int, int]) -> float | None:
        v1_id, v2_id = edge
        p1 = self.vertices[v1_id].pos
        p2 = self.vertices[v2_id].pos
        return segment_line_intersection(line, (p1, p2))

    def _get_plane_containing_the_two_vertices(self, vid1: int, vid2: int) -> Tuple[int, int] | None:
        # Look for exactly one plane holding two different faces -- one
        # touching each vertex -- where the line through vid1/vid2 crosses
        # both faces. Returns a split edge from cutting one of those faces,
        # or None if no such single plane exists.
        self._check_vid_exists(vid1)
        self._check_vid_exists(vid2)

        line = self._get_line_equation((vid1, vid2))

        def _face_crossed_by_line(fid: int) -> bool:
            """Return True when the line intersects the face boundary in at least two points."""
            face = self.faces[fid]
            intersections: list[np.ndarray] = []

            def _add_point(point: np.ndarray) -> None:
                for existing in intersections:
                    if np.linalg.norm(existing - point) <= EPS:
                        return
                intersections.append(point)

            for half_edge in self._walk_face_half_edges(face):
                if half_edge.origin is None or half_edge.next is None or half_edge.next.origin is None:
                    continue

                a_id = half_edge.origin.id
                b_id = half_edge.next.origin.id
                a_pos = self.vertices[a_id].pos
                b_pos = self.vertices[b_id].pos
                a_side = point_side_to_line(a_pos, line)
                b_side = point_side_to_line(b_pos, line)

                # Entire boundary segment on the line contributes both endpoints.
                if a_side == 0 and b_side == 0:
                    _add_point(a_pos)
                    _add_point(b_pos)
                    continue

                if a_side == 0:
                    _add_point(a_pos)
                    continue

                ratio = segment_line_intersection(line, (a_pos, b_pos))
                if ratio is None:
                    continue

                if abs(ratio) <= EPS:
                    _add_point(a_pos)
                elif abs(ratio - 1) <= EPS:
                    _add_point(b_pos)
                elif 0 < ratio < 1:
                    _add_point(a_pos + ratio * (b_pos - a_pos))

            return len(intersections) >= 2

        faces_with_vid1 = {fid for fid, face in self.faces.items() if vid1 in self._face_vids(fid)}
        faces_with_vid2 = {fid for fid, face in self.faces.items() if vid2 in self._face_vids(fid)}

        candidate_planes = []
        chosen_faces_by_plane = {}

        for pid, plane in self.planes.items():
            fids1 = [fid for fid in plane.face_ids if fid in faces_with_vid1]
            fids2 = [fid for fid in plane.face_ids if fid in faces_with_vid2]

            found_pair = None
            for fid1 in fids1:
                for fid2 in fids2:
                    if fid1 == fid2:
                        continue
                    if _face_crossed_by_line(fid1) and _face_crossed_by_line(fid2):
                        found_pair = (fid1, fid2)
                        break
                if found_pair is not None:
                    break

            if found_pair is not None:
                candidate_planes.append(pid)
                chosen_faces_by_plane[pid] = found_pair

        if len(candidate_planes) != 1:
            return None

        pid = candidate_planes[0]
        fid1, fid2 = chosen_faces_by_plane[pid]

        split_edge = self._cut_face_along_line(fid1, (vid1, vid2))
        if split_edge is not None:
            return split_edge

        return self._cut_face_along_line(fid2, (vid1, vid2))

    def _get_higher_face_of_edge(self, edge_tuple: Tuple[int, int]) -> int:
        edge = self.half_edges[edge_tuple]
        face1 = edge.face
        face2 = edge.twin.face
        # BFS from face1's plane toward face2's plane in the layering graph:
        # reaching it means face1's plane is above, so face1 is the higher one.
        pid1 = face1.plane_id
        pid2 = face2.plane_id
        visited = set()
        to_visit = [pid1]
        while to_visit:
            pid = to_visit.pop()
            if pid in visited:
                continue
            visited.add(pid)
            if pid == pid2:
                return face1.id
            if self.layers.has_node(pid):
                for neighbor in self.layers.neighbors(pid):
                    if neighbor not in visited:
                        to_visit.append(neighbor)
        return face2.id

    def _find_edge_on_line_between_two_vertices(self, edge_tuple: Tuple[int, int]) -> Tuple[int, int]:
        vertex_positions = {vid: vertex.pos for vid, vertex in self.vertices.items()}
        linear_paths = find_linear_paths_nx(vertex_positions, list(self.half_edges.keys()), edge_tuple[0], edge_tuple[1])
        if len(linear_paths) == 0:
            raise ValueError(f"No linear crease found between vertices {edge_tuple}.")
        edge = tuple(sorted(linear_paths[0][:2]))
        return edge
