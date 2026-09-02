from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np

from .config import EPS
from .DCEL import Face
from .utils import (
    get_line_equation,
    point_side_to_line,
    reflect_point,
    segment_line_intersection,
)


class OrigamiFoldTraversalMixin:
    """Working out which faces move when folding along an edge, and applying M/V edge types."""

    def _set_edge_type(self, edge: tuple[int, int], edge_type: str) -> None:
        self.half_edges[(edge[0], edge[1])].type = edge_type
        self.half_edges[(edge[1], edge[0])].type = edge_type

    def _get_faces_of_edge(
        self,
        edge: tuple[int, int],
        folded_fid: int | None = None,
    ) -> tuple[Face | None, Face | None]:
        half_edge = self.half_edges[edge]
        face = half_edge.face
        twin_face = half_edge.twin.face if half_edge.twin is not None else None

        if folded_fid is None:
            return face, twin_face

        if face is not None and face.id == folded_fid:
            return face, twin_face
        if twin_face is not None and twin_face.id == folded_fid:
            return twin_face, face

        raise ValueError(f"Face {folded_fid} is not adjacent to edge {edge}.")

    def _is_face_above_face(self, upper_face: Face, lower_face: Face) -> bool:
        upper_pid = upper_face.plane_id
        lower_pid = lower_face.plane_id

        if upper_pid == lower_pid:
            return False
        if nx.has_path(self.layers, upper_pid, lower_pid):
            return True
        if nx.has_path(self.layers, lower_pid, upper_pid):
            return False

        raise ValueError(
            f"Could not determine the stacking order between faces {upper_face.id} and {lower_face.id}."
        )

    def _get_fold_edge_type(self, staying_face_orientation: int, folded_face_ends_above: bool) -> str:
        if folded_face_ends_above:
            return "V" if staying_face_orientation == 0 else "M"
        return "M" if staying_face_orientation == 0 else "V"

    def _update_fold_edges_type(
        self,
        fids_to_fold_vs_fold_edge: dict[int, tuple[int, int] | None],
        *,
        skip_internal_moving_edges: bool = False,
    ) -> None:
        visited_edges = set()
        folded_fids = set(fids_to_fold_vs_fold_edge)

        for folded_fid, edge in fids_to_fold_vs_fold_edge.items():
            if edge is None:
                continue

            normalized_edge = tuple(sorted(edge))
            if normalized_edge in visited_edges:
                continue
            visited_edges.add(normalized_edge)

            current_type = self.half_edges[edge].type
            folded_face, staying_face = self._get_faces_of_edge(edge, folded_fid=folded_fid)
            both_sides_moving = (
                skip_internal_moving_edges
                and staying_face is not None
                and staying_face.id in folded_fids
            )

            if both_sides_moving and self._is_fold_type(current_type):
                continue

            if self._is_fold_type(current_type):
                self._set_edge_type(edge, "F")
                continue

            if current_type != "F":
                raise ValueError(f"Edge {edge} has unsupported fold type {current_type}.")

            if staying_face is None:
                raise ValueError(f"Edge {edge} does not have a staying face to determine fold type.")

            folded_face_ends_above = self._is_face_above_face(folded_face, staying_face)
            staying_face_orientation = 1 - staying_face.orientation if both_sides_moving else staying_face.orientation
            new_type = self._get_fold_edge_type(staying_face_orientation, folded_face_ends_above)
            self._set_edge_type(edge, new_type)

    def _swap_mountain_valley_edges(self) -> None:
        visited_edges = set()

        for edge in list(self.half_edges.keys()):
            normalized_edge = tuple(sorted(edge))
            if normalized_edge in visited_edges:
                continue
            visited_edges.add(normalized_edge)

            edge_type = self.half_edges[edge].type
            if edge_type == "M":
                self._set_edge_type(edge, "V")
            elif edge_type == "V":
                self._set_edge_type(edge, "M")

    def _get_face_to_start_fold_from(self, edge: tuple[int, int], side: int) -> int:
        if edge in self.half_edges:
            self.root_logger.debug(f"Edge {edge} exists in half_edges")
            initial_face_to_fold, _ = self._get_edge_faces_by_side(edge, side)
        else:
            self.root_logger.debug(f"Edge {edge} not in half_edges, splitting face first")
            face_to_split = self._get_face_containing_the_two_vertices(edge[0], edge[1])
            initial_face_to_fold, _ = self._split_face(face_to_split, edge, side)
        return initial_face_to_fold

    def _is_face_on_side_of_edge(self, fid: int, edge: tuple[int, int], side: int) -> bool:
        line = self._get_line_equation(edge)
        face = self.faces[fid]
        curr_edge = face.edge
        
        while True:
            vid = curr_edge.origin.id
            curr_tuple = (curr_edge.origin.id, curr_edge.next.origin.id)
            if curr_tuple == edge or tuple(reversed(curr_tuple)) == edge:
                pass
            else:
                vid_side = self._get_vid_side(vid, line)
                if vid_side == side:
                    return True
            curr_edge = curr_edge.next
            if curr_edge == face.edge:
                break
        return False

    def _get_fold_seed_faces(self, initial_fid: int, edge: tuple[int, int], side: int) -> set[int]:
        self.root_logger.debug(f"Collecting fold seed faces from initial face {initial_fid} along edge {edge} on side {side}")
        # Seed only the face that is definitely on the moving side. Higher
        # stacked faces are added lazily after each lower face is cut, so the
        # overlap test is performed against the actual moving subface rather
        # than against the unsplit parent face.
        seed_faces = {initial_fid}
        self.root_logger.debug(f"Fold seed faces: {seed_faces}")
        return seed_faces

    def _get_overlapping_faces_above(self, fid: int, edge: tuple[int, int], side: int) -> set[int]:
        overlapping_faces = set()
        current_pid = self.faces[fid].plane_id

        for pid in self._get_pids_above_pid(current_pid):
            for other_fid in self.planes[pid].face_ids:
                if not self._is_face_on_side_of_edge(other_fid, edge, side):
                    continue
                if self._faces_overlap(fid, other_fid):
                    overlapping_faces.add(other_fid)

        return overlapping_faces

    def _get_fold_side_neighbor_faces(
        self,
        fid: int,
        edge: tuple[int, int],
        side: int,
        include_shared_edges: bool = False,
    ) -> set[int] | dict[int, tuple[int, int]]:
        # Faces adjacent to fid across edges that sit on the given side of the fold line.
        neighbors = {} if include_shared_edges else set()
        line = self._get_line_equation(edge)
        face = self.faces[fid]

        for curr_edge in self._walk_face_half_edges(face):
            if curr_edge.origin is None or curr_edge.next is None or curr_edge.next.origin is None:
                continue

            vid = curr_edge.origin.id
            next_vid = curr_edge.next.origin.id
            curr_tuple = (vid, next_vid)
            vid_side = self._get_vid_side(vid, line)
            next_vid_side = self._get_vid_side(next_vid, line)

            if vid_side == 0 and next_vid_side == 0:
                self.root_logger.debug(f"Edge {curr_tuple} lies on the fold line, not traversing across it")
                continue

            if vid_side not in (side, 0) and next_vid_side not in (side, 0):
                continue

            twin_face = curr_edge.twin.face if curr_edge.twin is not None else None
            if twin_face is not None:
                if include_shared_edges:
                    neighbors.setdefault(twin_face.id, curr_tuple)
                else:
                    neighbors.add(twin_face.id)

        return neighbors

    def _face_intersects_line_in_unfolded_state(
        self,
        fid: int,
        edge: tuple[int, int],
        face_transforms: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> bool:
        from .crease_pattern import CreasePattern

        if edge not in self.half_edges:
            return False

        reference_half_edge = self.half_edges[edge]
        reference_face = reference_half_edge.face or (
            reference_half_edge.twin.face if reference_half_edge.twin is not None else None
        )
        if reference_face is None:
            return False

        reference_matrix, reference_offset = face_transforms[reference_face.id]
        p1 = CreasePattern._apply_transform(reference_matrix, reference_offset, self.vertices[edge[0]].pos)
        p2 = CreasePattern._apply_transform(reference_matrix, reference_offset, self.vertices[edge[1]].pos)
        line = get_line_equation(p1, p2)

        face_matrix, face_offset = face_transforms[fid]
        intersections: list[np.ndarray] = []

        def add_point(point: np.ndarray) -> None:
            for existing in intersections:
                if np.linalg.norm(existing - point) <= EPS:
                    return
            intersections.append(point)

        for half_edge in self._walk_face_half_edges(self.faces[fid]):
            if half_edge.origin is None or half_edge.next is None or half_edge.next.origin is None:
                continue

            start = CreasePattern._apply_transform(face_matrix, face_offset, half_edge.origin.pos)
            end = CreasePattern._apply_transform(face_matrix, face_offset, half_edge.next.origin.pos)
            start_side = point_side_to_line(start, line)
            end_side = point_side_to_line(end, line)

            if start_side == 0 and end_side == 0:
                add_point(start)
                add_point(end)
                continue

            if start_side == 0:
                add_point(start)
                continue

            ratio = segment_line_intersection(line, (start, end))
            if ratio is None:
                continue
            if abs(ratio) <= EPS:
                add_point(start)
            elif abs(ratio - 1) <= EPS:
                add_point(end)
            elif 0 < ratio < 1:
                add_point(start + ratio * (end - start))

        return len(intersections) >= 2

    def _face_has_isolated_opposite_side_vertex(self, fid: int, edge: tuple[int, int], side: int) -> bool:
        """Return True if any vertex strictly on the non-fold side of the fold line
        has only boundary (B) edges incident to it within the face — meaning it is
        a pure-boundary vertex that cannot be moved rigidly without crossing the fold line."""
        line_eq = self._get_line_equation(edge)
        opposite_side = -side
        for he in self._walk_face_half_edges(self.faces[fid]):
            if he.origin is None or he.next is None or he.next.origin is None:
                continue
            vid = he.origin.id
            if self._get_vid_side(vid, line_eq) != opposite_side:
                continue
            has_fold_connection = False
            for scan_he in self._walk_face_half_edges(self.faces[fid]):
                if scan_he.origin is None or scan_he.next is None or scan_he.next.origin is None:
                    continue
                if scan_he.origin.id == vid or scan_he.next.origin.id == vid:
                    if scan_he.type != 'B' or (scan_he.twin is not None and scan_he.twin.face is not None):
                        has_fold_connection = True
                        break
            if not has_fold_connection:
                return True
        return False

    def _get_fids_to_fold(
        self,
        initial_fid: int,
        edge: tuple[int, int],
        side: int,
        *,
        traverse_stacked_faces: bool = False,
        split_intersected_faces: bool = True,
        rigidly_move_attached_neighbors: bool = False,
    ) -> dict[int, tuple[int, int] | None]:
        # Traverse out from the seed faces and collect every face that must move
        # for this fold, mapped to the edge each one was cut along (or None).
        seed_faces = self._get_fold_seed_faces(initial_fid, edge, side)
        self.root_logger.debug(f"Traversing fold chain starting from seeds {seed_faces}")

        fids_to_fold = dict()
        visited = set()
        to_visit = list(seed_faces)
        queued_entry_edges = {fid: None for fid in seed_faces} if rigidly_move_attached_neighbors else None
        unfolded_face_transforms = None

        def queue_face(fid_to_queue: int, entry_edge: tuple[int, int] | None = None) -> None:
            if fid_to_queue in visited:
                return
            if queued_entry_edges is not None:
                if fid_to_queue not in queued_entry_edges:
                    queued_entry_edges[fid_to_queue] = entry_edge
                elif queued_entry_edges[fid_to_queue] is None and entry_edge is not None:
                    queued_entry_edges[fid_to_queue] = entry_edge
            to_visit.append(fid_to_queue)

        while to_visit:
            fid = to_visit.pop()
            if fid not in self.faces:
                self.root_logger.debug(f"Skipping stale face {fid}")
                continue
            if fid in visited:
                continue

            self.root_logger.debug(f"Processing face {fid}")
            visited.add(fid)
            entry_edge = queued_entry_edges.pop(fid, None) if queued_entry_edges is not None else None
            entered_via_attached_fold = (
                rigidly_move_attached_neighbors
                and entry_edge is not None
                and entry_edge in self.half_edges
                and self._is_fold_type(self.half_edges[entry_edge].type)
            )
            stacked_neighbor_fids = set()

            split_edge = self._get_face_edge_on_line(fid, edge)
            if split_edge is not None:
                self.root_logger.debug(
                    f"Face {fid} already has boundary edge {split_edge} on fold line {edge}"
                )
            elif split_intersected_faces:
                if entered_via_attached_fold:
                    if unfolded_face_transforms is None:
                        from .crease_pattern import CreasePattern
                        unfolded_face_transforms = CreasePattern._get_unfolded_face_transforms(self)
                if entered_via_attached_fold and not self._face_has_isolated_opposite_side_vertex(fid, edge, side) and not self._face_intersects_line_in_unfolded_state(
                    fid,
                    edge,
                    unfolded_face_transforms,
                ):
                    self.root_logger.debug(
                        f"Face {fid} is attached through existing fold edge {entry_edge}; "
                        "moving it rigidly instead of cutting a new crease through it"
                    )
                else:
                    split_edge = self._cut_face_along_line(fid, edge)
                    self.root_logger.debug(f"The fold edge {edge} crossed face {fid} at edge {split_edge}")
            else:
                self.root_logger.debug(
                    f"Keeping face {fid} whole while traversing existing crease line {edge}"
                )

            if split_edge is None:
                if not self._is_face_on_side_of_edge(fid, edge, side):
                    continue
                self.root_logger.debug(f"Edge {edge} does not cross face {fid}, but it is on the folding side {side}")
                folded_fid = fid
                fold_edge = None
            else:
                if self._face_has_edge(self.faces[fid], split_edge):
                    self.root_logger.debug(
                        f"Face {fid} already has boundary edge {split_edge}; "
                        "keeping the current face instead of traversing across the crease"
                    )
                    folded_fid = fid
                    split_half_edge = self.half_edges.get(split_edge)
                    twin_face = split_half_edge.twin.face if split_half_edge is not None and split_half_edge.twin is not None else None
                    edge_separates_two_faces = (
                        split_half_edge is not None
                        and split_half_edge.type in {"F", "M", "V"}
                        and twin_face is not None
                    )
                    if edge_separates_two_faces:
                        fold_edge = split_edge
                    else:
                        self.root_logger.debug(
                            f"Face {fid} only touches fold line {edge} along boundary segment {split_edge}; "
                            "moving it rigidly without flattening across that segment"
                        )
                        fold_edge = None
                    if (
                        traverse_stacked_faces
                        and split_half_edge is not None
                        and self._is_fold_type(split_half_edge.type)
                        and twin_face is not None
                        and twin_face.id != fid
                        and self._is_face_on_side_of_edge(twin_face.id, edge, side)
                    ):
                        self.root_logger.debug(
                            f"Edge {split_edge} lies on the fold line; adding stacked face {twin_face.id} "
                            "to continue traversal through the folded side"
                        )
                        stacked_neighbor_fids.add(twin_face.id)
                else:
                    folded_fid, _ = self._split_face(fid, split_edge, side)
                    fold_edge = split_edge

            visited.add(folded_fid)
            fids_to_fold[folded_fid] = fold_edge

            if traverse_stacked_faces:
                for overlapping_fid in self._get_overlapping_faces_above(folded_fid, edge, side):
                    if overlapping_fid not in visited:
                        self.root_logger.debug(
                            f"Adding overlapping stacked face {overlapping_fid} above folded face {folded_fid}"
                        )
                        if rigidly_move_attached_neighbors:
                            queue_face(overlapping_fid)
                        else:
                            to_visit.append(overlapping_fid)

            if rigidly_move_attached_neighbors:
                neighbor_faces = self._get_fold_side_neighbor_faces(
                    folded_fid,
                    edge,
                    side,
                    include_shared_edges=True,
                ).items()
            else:
                neighbor_faces = (
                    (neighbor_fid, None)
                    for neighbor_fid in self._get_fold_side_neighbor_faces(folded_fid, edge, side)
                )

            for neighbor_fid, shared_edge in neighbor_faces:
                if neighbor_fid not in visited:
                    self.root_logger.debug(f"Adding neighboring folded-side face {neighbor_fid} to visit stack")
                    if rigidly_move_attached_neighbors:
                        queue_face(neighbor_fid, shared_edge)
                    else:
                        to_visit.append(neighbor_fid)

            for neighbor_fid in stacked_neighbor_fids:
                if neighbor_fid not in visited:
                    self.root_logger.debug(f"Adding stacked folded-side face {neighbor_fid} to visit stack")
                    if rigidly_move_attached_neighbors:
                        queue_face(neighbor_fid, split_edge)
                    else:
                        to_visit.append(neighbor_fid)

        return fids_to_fold

    def _flip_face_orientation(self, fids: Iterable[int]) -> None:
        for fid in fids:
            self.faces[fid].orientation = 1 - self.faces[fid].orientation

    def _reflect_vertex_across_line(
        self,
        fold_edge: tuple[int, int],
        fids_to_fold: Iterable[int],
        side: int,
    ) -> set[int]:
        line_eq = self._get_line_equation(fold_edge)
        visited_vertices = set()
        self.root_logger.debug(f"Reflecting {len(fids_to_fold)} faces across line {line_eq}")
        for fid in fids_to_fold:
            self.root_logger.debug(f"Processing face {fid} for reflection")
            face = self.faces[fid]
            curr_edge = face.edge
            while True:
                vid = curr_edge.origin.id
                vids_side = self._get_vid_side(vid, line_eq)
                if vid not in visited_vertices and vids_side != 0:
                    visited_vertices.add(vid)
                    new_pos = reflect_point(self.vertices[vid].pos, line_eq)
                    old_pos = self.vertices[vid].pos
                    self.vertices[vid].pos = new_pos
                    self.root_logger.debug(f"Reflected vertex {vid} from {old_pos} to {new_pos}")
                curr_edge = curr_edge.next
                if curr_edge == face.edge:
                    break
            self.root_logger.debug(f"Reflected {len(visited_vertices)} vertices")
        return visited_vertices

    def _handle_mixed_fold(
        self,
        fids_to_fold_vs_fold_edge: dict[int, tuple[int, int] | None],
        planes_to_fold: set[int],
    ) -> None:
        # Fix up layering and plane merges for faces whose fold edge was
        # already flat (pre-merged) before this fold started.
        visited_edges = set()
        for fid, fold_edge in fids_to_fold_vs_fold_edge.items():
            if fold_edge is not None and self.half_edges[fold_edge].type == 'F':
                normalized_edge = tuple(sorted(fold_edge))
                if normalized_edge in visited_edges:
                    continue
                visited_edges.add(normalized_edge)

                # Reverse relations to static planes
                plane_id = self.faces[fid].plane_id
                predecessors = list(self.layers.predecessors(plane_id))
                for neighbor in predecessors:
                    if neighbor not in planes_to_fold:
                        if self.layers.has_edge(neighbor, plane_id):
                            self.layers.remove_edge(neighbor, plane_id)
                            self.layers.add_edge(plane_id, neighbor)

                pid1 = self.faces[self.half_edges[fold_edge].face.id].plane_id
                pid2 = self.faces[self.half_edges[fold_edge].twin.face.id].plane_id
                if pid1 == pid2 or pid1 not in self.planes or pid2 not in self.planes:
                    continue
                plane1 = self.planes.pop(pid1)
                fids1 = plane1.face_ids
                for fid in fids1:
                    self.faces[fid].plane_id = pid2
                self.planes[pid2].face_ids.extend(plane1.face_ids)
                
                # Transfer pid1's external relations onto pid2, but do not recreate
                # an opposite ordering that pid2 already established during the mixed fold.
                predecessors = list(self.layers.predecessors(pid1))
                successors = list(self.layers.successors(pid1))

                if self.layers.has_edge(pid1, pid2):
                    self.layers.remove_edge(pid1, pid2)
                if self.layers.has_edge(pid2, pid1):
                    self.layers.remove_edge(pid2, pid1)

                for predecessor in predecessors:
                    if predecessor == pid2:
                        continue
                    if self.layers.has_edge(pid2, predecessor):
                        continue
                    if nx.has_path(self.layers, pid2, predecessor):
                        continue
                    self.layers.add_edge(predecessor, pid2)

                for successor in successors:
                    if successor == pid2:
                        continue
                    if self.layers.has_edge(successor, pid2):
                        continue
                    if nx.has_path(self.layers, successor, pid2):
                        continue
                    self.layers.add_edge(pid2, successor)

                self.layers.remove_node(pid1)

    def _fold_along_edge(
        self,
        edge: tuple[int, int],
        side: int,
        fold_id: str | int | None = None,
    ) -> set[int]:
        """Fold the paper along the specified edge in the given direction."""
        self.root_logger.debug(f"Starting fold along edge {edge} on side {side}")
        edge_preexisted_as_flat = edge in self.half_edges and self.half_edges[edge].type == "F"

        # Get the face that contains the edge, or the two vertices of the edge
        initial_face_to_fold = self._get_face_to_start_fold_from(edge, side)

        # Traverse, cut, and collect only the actual folded-side faces.
        fids_to_fold_vs_fold_edge = self._get_fids_to_fold(
            initial_face_to_fold,
            edge,
            side,
            traverse_stacked_faces=True,
            rigidly_move_attached_neighbors=edge_preexisted_as_flat,
        )
        fids_to_fold = fids_to_fold_vs_fold_edge.keys()
        self.root_logger.debug(f"Faces to fold: {fids_to_fold}")

        # Split all planes
        planes_to_fold_vs_origin_plane = self._get_planes_to_fold(fids_to_fold)
        planes_to_fold = set(planes_to_fold_vs_origin_plane.keys())
        self.root_logger.debug(f"Planes to fold: {planes_to_fold}")

        self._set_first_plane_at_the_top_of_the_current_stack(planes_to_fold)
        
        self._flip_relations_between_folded_planes(planes_to_fold)
        self._drop_conflicting_fold_boundary_relations(planes_to_fold)

        # Place the folded plane above its original plane
        self._place_folded_plane_above_original(planes_to_fold_vs_origin_plane)
        self._drop_conflicting_split_origin_relations(planes_to_fold_vs_origin_plane)
        self._drop_conflicting_fold_boundary_relations(planes_to_fold)
        self._update_fold_edges_type(fids_to_fold_vs_fold_edge, skip_internal_moving_edges=True)

        # Update the faces orientations
        self._flip_face_orientation(fids_to_fold)

        # Reflect the vertices of the faces that are folded
        reflected_vids = self._reflect_vertex_across_line(edge, fids_to_fold, side)
        self._place_folded_planes_above_newly_overlapped_planes(planes_to_fold)

        self._handle_mixed_fold(fids_to_fold_vs_fold_edge, planes_to_fold)
        self._prune()

        self.root_logger.debug(f"Fold complete.")

        if fold_id is not None:
            self.folds_information[fold_id] = {
                "fids_to_fold_vs_fold_edge": fids_to_fold_vs_fold_edge,
                "planes_to_fold": planes_to_fold,
                "reflected_vids": reflected_vids,
            }

        return set(fids_to_fold)
